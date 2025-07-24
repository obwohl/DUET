# ts_benchmark/baselines/duet/spliced_binned_pareto_standalone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

class SplicedBinnedPareto(Distribution):
    arg_constraints = {
        "lower_gp_xi": constraints.real, "lower_gp_beta": constraints.positive,
        "upper_gp_xi": constraints.real, "upper_gp_beta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False 
    
    def __init__(self, logits, lower_gp_xi, lower_gp_beta, upper_gp_xi, upper_gp_beta,
                 bins_lower_bound, bins_upper_bound, tail_percentile, validate_args=None):
        self.num_bins = logits.shape[-1]
        self.bins_lower_bound = bins_lower_bound
        self.bins_upper_bound = bins_upper_bound
        self.tail_percentile = tail_percentile
        self.logits = logits
        
        (lower_gp_xi, lower_gp_beta, upper_gp_xi, upper_gp_beta) = broadcast_all(
            lower_gp_xi, lower_gp_beta, upper_gp_xi, upper_gp_beta)
        self.lower_gp_xi = lower_gp_xi
        self.lower_gp_beta = lower_gp_beta
        self.upper_gp_xi = upper_gp_xi
        self.upper_gp_beta = upper_gp_beta
        
        batch_shape = self.lower_gp_xi.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        self._bin_width = (bins_upper_bound - bins_lower_bound) / self.num_bins
        self.bin_edges = torch.linspace(
            bins_lower_bound, bins_upper_bound, self.num_bins + 1, device=logits.device)

        self.lower_threshold = self._icdf_binned(torch.full_like(self.lower_gp_xi, self.tail_percentile))
        self.upper_threshold = self._icdf_binned(torch.full_like(self.upper_gp_xi, 1.0 - self.tail_percentile))

    def _icdf_binned(self, quantiles: torch.Tensor) -> torch.Tensor:
        # print(f"  [DEBUG] ==> Entering _icdf_binned, quantiles shape: {quantiles.shape}")
        # --- FIX: Complete rewrite of the function for robust broadcasting ---
        bin_probs = torch.softmax(self.logits, dim=-1)
        bin_cdfs = torch.cumsum(bin_probs, dim=-1)  # Shape: [B, H, n_bins]

        # Prepare CDFs for broadcasting.
        cdfs_bcast = bin_cdfs.unsqueeze(-2)  # Shape: [B, H, 1, n_bins]

        # Prepare quantiles for broadcasting, handling both 2D and 3D cases.
        if quantiles.dim() == 2:  # Case from __init__, shape [B, H]
            q_bcast = quantiles.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, H, 1, 1]
        else:  # Case from CRPS loss, shape [B, H, n_q]
            q_bcast = quantiles.unsqueeze(-1)  # Shape: [B, H, n_q, 1]

        # Comparison and sum to find bin indices.
        bin_indices = torch.sum(cdfs_bcast < q_bcast, dim=-1)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Now, `bin_indices` has the same number of dimensions as `bin_probs`
        prob_in_bin = torch.gather(bin_probs, -1, bin_indices)
        
        cdf_start_of_bin_padded = F.pad(bin_cdfs[..., :-1], (1, 0), 'constant', 0.0)
        cdf_start_of_bin = torch.gather(cdf_start_of_bin_padded, -1, bin_indices)

        # We need to squeeze the results of gather to match the shape of quantiles.
        numerator = quantiles - cdf_start_of_bin.squeeze(-1)
        safe_prob_in_bin = torch.clamp(prob_in_bin.squeeze(-1), min=1e-9)
        frac_in_bin = numerator / safe_prob_in_bin
        frac_in_bin = torch.clamp(frac_in_bin, 0.0, 1.0)
        
        # Indexing bin_edges requires a long tensor without the trailing dimension.
        lower_edge = self.bin_edges[bin_indices.squeeze(-1)]
        
        # --- KORREKTUR: Der entscheidende Bugfix ---
        # Die Position innerhalb des Bins muss mit der Breite des Bins skaliert werden.
        result = lower_edge + frac_in_bin * self._bin_width
        # print(f"  [DEBUG] <== Exiting _icdf_binned, result shape: {result.shape}")
        return result
    def icdf(self, quantiles: torch.Tensor) -> torch.Tensor:
        # Check the dimension of the input tensor. This is key to handling both cases.
        input_dim = quantiles.dim()

        # Unsqueeze parameters to allow broadcasting with multi-quantile input.
        lower_threshold = self.lower_threshold.unsqueeze(-1)
        upper_threshold = self.upper_threshold.unsqueeze(-1)
        lower_gp_xi = self.lower_gp_xi.unsqueeze(-1)
        lower_gp_beta = self.lower_gp_beta.unsqueeze(-1)
        upper_gp_xi = self.upper_gp_xi.unsqueeze(-1)
        upper_gp_beta = self.upper_gp_beta.unsqueeze(-1)

        # Use a consistent, broadcastable view of quantiles for tail calculations
        q_view = quantiles if input_dim > 2 else quantiles.unsqueeze(-1)

        q_adj_lower = q_view / (self.tail_percentile + 1e-9)
        is_near_zero_lower = torch.abs(lower_gp_xi) < 1e-6
        icdf_gpd_lower = lower_threshold - (lower_gp_beta / lower_gp_xi) * (torch.pow(q_adj_lower, -lower_gp_xi) - 1.0)
        icdf_exp_lower = lower_threshold + lower_gp_beta * torch.log(q_adj_lower)
        icdf_lower = torch.where(is_near_zero_lower, icdf_exp_lower, icdf_gpd_lower)

        q_adj_upper = (1.0 - q_view) / (self.tail_percentile + 1e-9)
        is_near_zero_upper = torch.abs(upper_gp_xi) < 1e-6
        icdf_gpd_upper = upper_threshold + (upper_gp_beta / upper_gp_xi) * (torch.pow(q_adj_upper, -upper_gp_xi) - 1.0)
        icdf_exp_upper = upper_threshold - upper_gp_beta * torch.log(q_adj_upper)
        icdf_upper = torch.where(is_near_zero_upper, icdf_exp_upper, icdf_gpd_upper)

        icdf_binned = self._icdf_binned(quantiles)
        # Unsqueeze the binned results to match the dimension of the tail results
        icdf_binned_bcast = icdf_binned if input_dim > 2 else icdf_binned.unsqueeze(-1)
        
        # Use broadcastable conditions
        in_lower_tail = quantiles < self.tail_percentile
        in_upper_tail = quantiles > (1.0 - self.tail_percentile)
        condition_view = in_lower_tail if input_dim > 2 else in_lower_tail.unsqueeze(-1)

        value = torch.where(condition_view, icdf_lower, icdf_binned_bcast)
        
        condition_view = in_upper_tail if input_dim > 2 else in_upper_tail.unsqueeze(-1)
        value = torch.where(condition_view, icdf_upper, value)
        
        # If the original input was 2D, squeeze the extra dimension from the output
        if input_dim == 2:
            value = value.squeeze(-1)

        finfo = torch.finfo(value.dtype)
        return torch.nan_to_num(value, nan=0.0, posinf=finfo.max, neginf=finfo.min)

    def log_prob(self, value: torch.Tensor, for_training: bool = False) -> torch.Tensor:
        value = value.expand(self.batch_shape)
        bin_indices = torch.clamp(torch.floor((value - self.bins_lower_bound) / self._bin_width), 0, self.num_bins - 1).long()
        bin_probs = torch.softmax(self.logits, dim=-1)
        log_prob_binned = torch.log(torch.gather(bin_probs, -1, bin_indices.unsqueeze(-1)).squeeze(-1) / self._bin_width + 1e-9)

        y = self.lower_threshold - value
        log1p_arg_lower = self.lower_gp_xi * y / self.lower_gp_beta
        log1p_arg_lower = torch.clamp(log1p_arg_lower, min=-1.0 + 1e-6)
        is_near_zero_lower = torch.abs(self.lower_gp_xi) < 1e-6
        log_prob_term_gpd_lower = - (1.0 + 1.0 / self.lower_gp_xi) * torch.log1p(log1p_arg_lower)
        log_prob_term_exp_lower = -y / self.lower_gp_beta
        log_prob_term_lower = torch.where(is_near_zero_lower, log_prob_term_exp_lower, log_prob_term_gpd_lower)
        log_prob_lower = (torch.log(torch.tensor(self.tail_percentile, device=value.device) + 1e-9) - torch.log(self.lower_gp_beta) + log_prob_term_lower)
        
        z = value - self.upper_threshold
        log1p_arg_upper = self.upper_gp_xi * z / self.upper_gp_beta
        log1p_arg_upper = torch.clamp(log1p_arg_upper, min=-1.0 + 1e-6)
        is_near_zero_upper = torch.abs(self.upper_gp_xi) < 1e-6
        log_prob_term_gpd_upper = - (1.0 + 1.0 / self.upper_gp_xi) * torch.log1p(log1p_arg_upper)
        log_prob_term_exp_upper = -z / self.upper_gp_beta
        log_prob_term_upper = torch.where(is_near_zero_upper, log_prob_term_exp_upper, log_prob_term_gpd_upper)
        log_prob_upper = (torch.log(torch.tensor(self.tail_percentile, device=value.device) + 1e-9) - torch.log(self.upper_gp_beta) + log_prob_term_upper)

        in_lower_tail = value < self.lower_threshold
        in_upper_tail = value > self.upper_threshold
        log_p = torch.where(in_lower_tail, log_prob_lower, torch.where(in_upper_tail, log_prob_upper, log_prob_binned))
        
        if for_training:
            in_any_tail = torch.logical_or(in_lower_tail, in_upper_tail)
            log_p = torch.where(in_any_tail, log_p + log_prob_binned.detach(), log_p)
            
        return log_p

class ProjectionResidualBlock(nn.Module):
    """A single residual block for the projection head, mimicking a Transformer's FFN."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.norm(x + residual)
        return x

class MLPProjectionHead(nn.Module):
    """A non-linear projection head composed of residual blocks."""
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers

        if self.num_layers == 0:
            # Fallback to the original simple linear layer for num_layers=0
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.residual_blocks = nn.ModuleList(
                [ProjectionResidualBlock(d_model=in_features, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )
            self.final_layer = nn.Linear(in_features, out_features)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers > 0:
            for block in self.residual_blocks:
                x = block(x)
            return self.final_layer(x)
        return self.projection(x)

class SplicedBinnedParetoOutput:
    def __init__(self, num_bins: int, bins_lower_bound: float, bins_upper_bound: float, tail_percentile: float,
                 projection_head_layers: int = 0, projection_head_dim_factor: int = 2, projection_head_dropout: float = 0.1):
        self.num_bins = num_bins
        self.bins_lower_bound = bins_lower_bound
        self.bins_upper_bound = bins_upper_bound
        self.tail_percentile = tail_percentile
        self.args_dim = num_bins + 4
        self.projection_head_layers = projection_head_layers
        self.projection_head_dim_factor = projection_head_dim_factor
        self.projection_head_dropout = projection_head_dropout

    def get_args_proj(self, in_features: int) -> nn.Module:
        # `in_features` is the model's `d_model`
        hidden_dim = max(self.args_dim, in_features // self.projection_head_dim_factor)
        return MLPProjectionHead(
            in_features=in_features,
            out_features=self.args_dim,
            hidden_dim=hidden_dim,
            num_layers=self.projection_head_layers,
            dropout=self.projection_head_dropout
        )

    def distribution(self, distr_args: torch.Tensor) -> "SplicedBinnedPareto":
        logits_raw = distr_args[..., :self.num_bins]
        lower_gp_xi_raw, lower_gp_beta_raw, upper_gp_xi_raw, upper_gp_beta_raw = [
            distr_args[..., i] for i in range(self.num_bins, self.num_bins + 4)
        ]
        
        logits = 10.0 * torch.tanh(logits_raw)
        BETA_FLOOR = 0.1
        lower_gp_beta = F.softplus(lower_gp_beta_raw) + BETA_FLOOR
        upper_gp_beta = F.softplus(upper_gp_beta_raw) + BETA_FLOOR
        lower_gp_xi = 1.5 * torch.tanh(lower_gp_xi_raw) + 0.5
        upper_gp_xi = 1.5 * torch.tanh(upper_gp_xi_raw) + 0.5

        return SplicedBinnedPareto(
            logits=logits, lower_gp_xi=lower_gp_xi, lower_gp_beta=lower_gp_beta,
            upper_gp_xi=upper_gp_xi, upper_gp_beta=upper_gp_beta,
            bins_lower_bound=self.bins_lower_bound,
            bins_upper_bound=self.bins_upper_bound,
            tail_percentile=self.tail_percentile,
        )