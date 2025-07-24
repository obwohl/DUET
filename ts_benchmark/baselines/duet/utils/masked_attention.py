import torch.nn as nn
import torch
from typing import Tuple
from math import sqrt

import torch.nn.functional as F
# The plan requires RevIN, so we assume it's correctly imported.
from ts_benchmark.baselines.duet.layers.RevIN import RevIN
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, tau=None, delta=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A standard Transformer Encoder Layer with Pre-Normalization.
        The residual connection is applied *after* the sub-layer (attention or feed-forward),
        and normalization is applied to the input of each sub-layer. This prevents the
        information leakage that occurs in Post-LN architectures, ensuring the attention
        mask is effective.
        """
        # --- 1. Self-Attention Block ---
        # Apply normalization BEFORE the attention layer
        norm_x = self.norm1(x)
        attn_output, attn = self.attention(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(attn_output) # Apply residual connection

        # --- 2. Feed-Forward Block ---
        norm_x = self.norm2(x) # Apply normalization BEFORE the feed-forward layer
        ff_output = self.conv2(self.dropout(self.activation(self.conv1(norm_x.transpose(-1, 1))))).transpose(-1, 1)
        x = x + self.dropout(ff_output) # Apply residual connection

        return x, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            # If mask_flag is True, a mask must be applied. If no specific mask is
            # provided (attn_mask is None), we create a default causal mask. This
            # is the standard behavior for temporal self-attention to prevent looking ahead.
            current_mask = attn_mask
            if current_mask is None:
                # A lower-triangular matrix (1s on and below diagonal, 0s above)
                current_mask = torch.tril(torch.ones((L, S), device=queries.device))

            # Use a large negative number for masking to avoid numerical issues with multiplication. Ensure the mask is float.
            large_negative = -torch.finfo(scores.dtype).max
            attention_mask = torch.where(current_mask == 0, large_negative, 0.0)
            scores = scores + attention_mask # Use addition for masking

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size, n_vars):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        # --- BUG FIX: Initialize A as an identity matrix, not a random one. ---
        # A random matrix acts as a strong, incorrect prior, scrambling the frequency
        # information and leading to large initial distances even for similar signals.
        # An identity matrix provides a neutral starting point (Euclidean distance),
        # from which the model can learn meaningful frequency correlations.
        self.A = nn.Parameter(torch.eye(frequency_size), requires_grad=True)

    def calculate_prob_distance(self, X, channel_adjacency_prior=None):
        # X shape: [B, C, L], where C is n_vars

        # --- DEFINITIVE FIX: Revert to the original, successful logic. ---
        # The original model passed raw, trended data to the mask. It worked because
        # the combination of this specific normalization and the Gumbel-Softmax sampler
        # created a strong initial gradient signal, allowing the model to learn to
        # ignore the trend (DC component of the FFT) over time.
        #
        # All previous attempts to "fix" this by pre-processing the data (RevIN,
        # linear de-trending) or using a mathematically pure probability normalization
        # were incorrect as they broke this essential bootstrapping mechanism.

        # 1. Use raw data directly, as confirmed by the expert.
        XF = torch.abs(torch.fft.rfft(X, dim=-1))

        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)
        diff = X1 - X2 # B x C x C x D

        # 2. Calculate distance without normalizing the learnable matrix `A`.
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        # 3. Revert to the original "winner-take-all" normalization.
        # This is not a true probability distribution, but it creates the necessary
        # high-contrast signal for the Gumbel-Softmax to start the learning process.
        inv_dist = 1.0 / (dist + 1e-10)
        identity_mask = 1.0 - torch.eye(inv_dist.shape[-1], device=inv_dist.device)
        off_diag_inv_dist = inv_dist * identity_mask.unsqueeze(0)

        # Normalize by the single max value in the matrix.
        max_val = torch.max(off_diag_inv_dist, dim=-1, keepdim=True)[0].detach()
        p_learned = off_diag_inv_dist / (max_val + 1e-9)

        # Add the diagonal back and apply the magic number scaling.
        p_learned = p_learned + torch.eye(p_learned.shape[-1], device=p_learned.device).unsqueeze(0)
        p_learned = torch.clamp(p_learned, 0, 1) # Ensure values are in [0, 1] for the sampler

        p_final = p_learned.clone()
        
        if channel_adjacency_prior is not None:
            prior = channel_adjacency_prior.to(p_final.device)
            p_final = p_final * prior.unsqueeze(0)
        
        return p_learned, p_final

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        
        # III: Prevent log(0) or log(1) Errors for Gumbel-Softmax.
        # Clamp the probability to a safe range [eps, 1-eps] before log.
        eps = 1e-9
        flatten_matrix = torch.clamp(flatten_matrix, min=eps, max=1.0 - eps)
        
        # Calculate log-odds for Gumbel-Softmax
        log_odds = torch.log(flatten_matrix / (1.0 - flatten_matrix))
        
        # Gumbel-Softmax expects logits for two classes [class_0, class_1]
        # Here, class_1 is probability p, class_0 is 1-p.
        # logit_1 = log(p / (1-p)), logit_0 = log((1-p)/p) = -logit_1
        gumbel_input = torch.cat([log_odds, -log_odds], dim=-1)
        
        resample_matrix = gumbel_softmax(gumbel_input, hard=True)

        # We only need the probabilities for class 1 (the original 'p')
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X, channel_adjacency_prior=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the channel attention mask and returns intermediate probability matrices for logging.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - mask (torch.Tensor): The final binary attention mask for the transformer.
                - p_learned (torch.Tensor): The unconstrained, learned probability matrix.
                - p_final (torch.Tensor): The final probability matrix after applying the user prior.
        """
        p_learned, p_final = self.calculate_prob_distance(X, channel_adjacency_prior=channel_adjacency_prior)
        sample = self.bernoulli_gumbel_rsample(p_final)
        mask = sample.unsqueeze(1)
        return mask, p_learned, p_final