# ts_benchmark/baselines/duet/models/duet_prob_model.py
# (BASIEREND AUF DEM ORIGINALEN DUETMODEL, UMGEBAUT FÜR PROBABILISTISCHE VORHERSAGE)

from typing import Dict
import torch
import torch.nn as nn
from einops import rearrange

# === CORE-KOMPONENTEN VON DUET ===
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer

# === PROBABILISTISCHE KOMPONENTEN ===
from ts_benchmark.baselines.duet.spliced_binned_pareto_standalone import SplicedBinnedParetoOutput, MLPProjectionHead

# === HELPER-KLASSEN FÜR VERTEILUNGEN (aus deiner alten duet_prob_model.py kopiert) ===

class PerChannelDistribution:
    """ Hält eine Verteilung für jeden Kanal/jede Variable. """
    def __init__(self, distributions_dict: Dict[str, torch.distributions.Distribution], channel_order: list):
        self.distributions = distributions_dict
        self.channel_order = channel_order

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Erwartet value in [B, N_vars, H]
        log_probs_list = []
        for i, channel_name in enumerate(self.channel_order):
            dist = self.distributions[channel_name]
            channel_value = value[:, i, :]
            log_probs_list.append(dist.log_prob(channel_value))
        return torch.stack(log_probs_list, dim=1)

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # Gibt [B, H, N_vars] zurück
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, device=next(iter(self.distributions.values())).logits.device)

        quantiles_list = []
        for channel_name in self.channel_order:
            dist = self.distributions[channel_name]
            B, H, *_ = dist.batch_shape
            
            # Broadcaste q auf die richtige Dimension für die Verteilung
            if q.numel() > 1:
                # Fall: Mehrere Quantile (z.B. für CRPS oder Plotting)
                q_bcast = q.reshape(1, 1, -1).expand(B, H, -1)
            else:
                # Fall: Einzelnes Quantil
                q_bcast = q.expand(B, H)
            
            quantiles_list.append(dist.icdf(q_bcast))
            
        return torch.stack(quantiles_list, dim=2)

class DenormalizingDistribution:
    """ 
    Wrapper für die Denormalisierung. 
    Nimmt eine Basis-Verteilung auf normalisierten Daten und die Statistik (mean, std)
    und gibt eine Verteilung zurück, deren Samples (z.B. via icdf) auf der Originalskala liegen.
    """
    def __init__(self, base_distribution: PerChannelDistribution, stats: torch.Tensor):
        self.base_dist = base_distribution
        # stats hat die Form: [B, N_vars, 2]
        # self.mean, self.std bekommen die Form: [B, 1, N_vars] für Broadcasting
        self.mean = stats[:, :, 0].unsqueeze(1)
        STD_FLOOR = 1e-6 # Sicherheits-Floor für die Standardabweichung
        self.std = torch.clamp(stats[:, :, 1], min=STD_FLOOR).unsqueeze(1)

    @property
    def batch_shape(self):
        # Definiert die "Größe" der Verteilung
        base_shape = self.base_dist.distributions[self.base_dist.channel_order[0]].batch_shape
        return torch.Size([base_shape[0], base_shape[1], len(self.base_dist.channel_order)])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Erwartet `value` in [B, H, N_vars]
        value_norm = (value - self.mean) / self.std
        # base_dist.log_prob erwartet [B, N_vars, H], also permutieren
        log_p = self.base_dist.log_prob(value_norm.permute(0, 2, 1))
        # Korrekturterm (Log-Determinante der Jacobi-Matrix der Transformation)
        log_det_jacobian = torch.log(self.std).permute(0, 2, 1)
        return log_p - log_det_jacobian

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        # base_dist.icdf gibt normalisierte Werte zurück.
        # Die Form ist [B, H, N_vars] für ein einzelnes q,
        # und [B, H, N_vars, N_quantiles] für mehrere q's.
        value_norm = self.base_dist.icdf(q)

        # Hole die Formen für die Denormalisierung
        std_for_bcast = self.std
        mean_for_bcast = self.mean
        
        # Wenn value_norm eine extra Quantil-Dimension hat,
        # müssen wir mean und std auch eine hinzufügen, damit Broadcasting funktioniert.
        if value_norm.dim() > self.std.dim():
            std_for_bcast = self.std.unsqueeze(-1) # Form wird [B, 1, N_vars, 1]
            mean_for_bcast = self.mean.unsqueeze(-1) # Form wird [B, 1, N_vars, 1]

        # Die Multiplikation funktioniert jetzt:
        # [B, H, N_vars, N_q] * [B, 1, N_vars, 1] -> [B, H, N_vars, N_q]
        value_orig = value_norm * std_for_bcast + mean_for_bcast
        return value_orig

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalisiert einen externen Wert (z.B. den Zielwert) mit den Statistiken dieser Verteilung.
        Erwartet `value` in [B, H, N_vars].
        """
        return (value - self.mean) / self.std

# === DAS NEUE, PROBABILISTISCHE DUET MODELL ===

class DUETProbModel(nn.Module): # Umbenannt von DUETModel
    def __init__(self, config):
        super(DUETProbModel, self).__init__()

        # --- Kernkomponenten von DUET (bleiben erhalten) ---
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.d_model = config.d_model
        self.horizon = config.horizon

        # Die Maske braucht die Anzahl der Variablen
        self.mask_generator = Mahalanobis_mask(config.seq_len, n_vars=self.n_vars)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        # --- Probabilistischer Kopf (ersetzt den alten `linear_head`) ---
        # Helfer, um die Dimensionen der Verteilungsparameter zu bekommen
        self.distr_output_helper = SplicedBinnedParetoOutput(
            num_bins=getattr(config, 'num_bins', 100),
            bins_lower_bound=-1e6, # Nur Platzhalter
            bins_upper_bound=1e6,  # Nur Platzhalter
            tail_percentile=getattr(config, 'tail_percentile', 0.05)
        )
        
        # --- DEFINITIVE FIX: Use Per-Channel Projection Heads ---
        # The previous shared projection head was re-mixing channel information after
        # the Channel_transformer had carefully separated it. This created a difficult
        # optimization problem where the MLP had to learn two different functions
        # (one simple, one complex) with a single set of weights, leading to
        # conflicting gradients. By creating a dedicated head for each channel,
        # we ensure that the final distribution parameters for a channel are derived
        # ONLY from its own (masked) feature representation.
        self.channel_names = list(config.channel_bounds.keys())
        self.args_proj = nn.ModuleDict()
        in_features_per_channel = self.d_model
        out_features_per_channel = self.horizon * self.distr_output_helper.args_dim
        hidden_dim_factor = getattr(config, 'projection_head_dim_factor', 2)
        hidden_dim = max(self.distr_output_helper.args_dim, in_features_per_channel // hidden_dim_factor)

        for name in self.channel_names:
            self.args_proj[name] = MLPProjectionHead(
                in_features=in_features_per_channel,
                out_features=out_features_per_channel,
                hidden_dim=hidden_dim,
                num_layers=getattr(config, 'projection_head_layers', 0),
                dropout=getattr(config, 'projection_head_dropout', 0.1)
            )

        # --- NEW: Store the user-defined channel adjacency prior ---
        self.channel_adjacency_prior = getattr(config, 'channel_adjacency_prior', None)
        if self.channel_adjacency_prior is not None:
            # The prior can be passed as a list of lists, so we ensure it's a tensor.
            if not isinstance(self.channel_adjacency_prior, torch.Tensor):
                self.channel_adjacency_prior = torch.tensor(self.channel_adjacency_prior, dtype=torch.float32)

            # Basic validation to prevent common errors
            if self.channel_adjacency_prior.shape != (self.n_vars, self.n_vars):
                raise ValueError(
                    f"channel_adjacency_prior shape mismatch. "
                    f"Expected ({self.n_vars}, {self.n_vars}), but got {self.channel_adjacency_prior.shape}"
                )

        # --- PROPOSED ENHANCEMENT: Early-stage channel mixer ---
        # This Conv1d with kernel size 1 acts as a per-timestep, learnable linear transformation
        # across the channel dimension, providing some cross-channel context to the MoE block.
        self.pre_cluster_mixer = nn.Conv1d(in_channels=self.n_vars, out_channels=self.n_vars, kernel_size=1)
 
        # --- KORREKTUR: Verteilungs-Setup ---
        # Der Bug war hier: Die Verteilungs-Köpfe wurden mit den Grenzen der
        # Original-Daten initialisiert, obwohl das Modell auf normalisierten Daten
        # operiert. Dies führte zu einer doppelten Denormalisierung.
        #
        # Die Lösung: Wir initialisieren die Köpfe mit festen, sinnvollen Grenzen
        # für normalisierte Daten (z.B. [-10, 10]). Die `channel_bounds` aus der
        # Konfiguration werden hier nicht mehr benötigt, da die Denormalisierung
        # korrekt über den `DenormalizingDistribution`-Wrapper und die `stats`
        # aus der RevIN-Schicht erfolgt.
        self.distr_output_dict = {name: SplicedBinnedParetoOutput(
            num_bins=getattr(config, 'num_bins', 100), bins_lower_bound=-10.0, bins_upper_bound=10.0, tail_percentile=getattr(config, 'tail_percentile', 0.05)
        ) for name in self.channel_names}

    def forward(self, input_x: torch.Tensor):
        # Der forward-Pass gibt jetzt ein Verteilungsobjekt und den MoE-Loss zurück.
        # input_x: [Batch, SeqLen, NVars]
        
        # --- 1. Normaler Modell-Pfad ---
        # RevIN (mit subtract_last) wird direkt auf den Original-Input angewendet.
        # RevIN gibt die normalisierten Daten und die Statistiken (mean, std) zurück.
        x_for_main_model, stats = self.cluster.revin(input_x, 'norm')
        x_for_main_model = torch.nan_to_num(x_for_main_model)

        # --- PROPOSED ENHANCEMENT: Apply the early-stage channel mixing ---
        # Wir wenden den Mixer auf den Input für den Hauptpfad an.
        # x_for_main_model shape: [B, L, N]. Conv1d expects [B, N, L].
        x_mixed = self.pre_cluster_mixer(x_for_main_model.permute(0, 2, 1)).permute(0, 2, 1)
        # Add as a residual connection so the model can choose to use it or ignore it.
        x_for_cluster = x_for_main_model + x_mixed

        # 2. Zeitliche Mustererkennung mit MoE (Linear_extractor_cluster)
        if self.CI:
            # Behandle jeden Kanal unabhängig
            # Use the (potentially) mixed features as input
            channel_independent_input = rearrange(x_for_cluster, 'b l n -> (b n) l 1')
            reshaped_output, L_importance, avg_gate_weights_linear, avg_gate_weights_esn, expert_selection_counts = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input_x.shape[0])
        else:
            # If not channel-independent, the cluster gets the mixed features directly.
            temporal_feature, L_importance, avg_gate_weights_linear, avg_gate_weights_esn, expert_selection_counts = self.cluster(x_for_cluster)

        # 3. Kanalübergreifende Interaktion mit Channel-Transformer
        # temporal_feature ist [B, D_Model, N_Vars] -> umformen zu [B, N_Vars, D_Model]
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        
        # Initialisiere die Matrizen, die zurückgegeben werden sollen
        p_learned, p_final = None, None

        if self.n_vars > 1:
            # --- DESIGN-VERBESSERUNG: Die Maske wird auf den un-gemischten Daten berechnet. ---
            # Die Mahalanobis-Maske soll die *intrinsische* Ähnlichkeit der Signale
            # bewerten. Der `pre_cluster_mixer` vermischt die Kanäle, was diese
            # Messung "verschmutzen" und zu widersprüchlichen Gradienten führen kann.
            # Wir übergeben daher die reinen, normalisierten Daten (`x_for_main_model`) an die Maske.
            changed_input = rearrange(x_for_main_model, 'b l n -> b n l')

            # --- BUG FIX: Removed noise addition. ---
            # The noise, intended for a past version of the distance metric, breaks the
            # shift-invariance of the FFT's magnitude, preventing the mask from correctly
            # identifying time-shifted signals. The current implementation does not need it.
            
            channel_mask, p_learned, p_final = self.mask_generator(
                changed_input,
                channel_adjacency_prior=self.channel_adjacency_prior
            )
            channel_group_feature, _ = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)
        else:
            channel_group_feature = temporal_feature
            # Erstelle Dummy-Matrizen für den Fall mit einer einzelnen Variable, um die Signatur konsistent zu halten.
            if self.n_vars == 1:
                p_learned = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)
                p_final = torch.ones(input_x.shape[0], 1, 1, device=input_x.device)

        # 4. Erzeugung der Verteilungsparameter
        # channel_group_feature ist [B, N_Vars, D_Model]
        # --- DEFINITIVE FIX: Apply Per-Channel Heads ---
        all_distr_params = []
        for i, name in enumerate(self.channel_names):
            # Get features for this channel: [B, D_Model]
            channel_feature = channel_group_feature[:, i, :]
            
            # Pass through this channel's dedicated head
            channel_distr_params_flat = self.args_proj[name](channel_feature)
            
            # Prevent NaN/inf in the parameters
            channel_distr_params_flat = torch.nan_to_num(channel_distr_params_flat, nan=0.0, posinf=1e4, neginf=-1e4)

            # Reshape to [B, Horizon, N_Params]
            channel_distr_params = rearrange(
                channel_distr_params_flat,
                'b (h d) -> b h d',
                h=self.horizon,
                d=self.distr_output_helper.args_dim
            )
            all_distr_params.append(channel_distr_params)

        # Stack to get the final parameter tensor: [B, N_Vars, Horizon, N_Params]
        distr_params = torch.stack(all_distr_params, dim=1)

        # 5. Erstellung der finalen Verteilungsobjekte
        channel_distributions = {}
        for i, channel_name in enumerate(self.channel_names):
            distr_helper = self.distr_output_dict[channel_name]
            # Parameter für diesen Kanal: [B, H, N_Params]
            params_for_channel = distr_params[:, i, :, :]
            channel_distributions[channel_name] = distr_helper.distribution(params_for_channel)
            
        # Wrappe die Verteilungen in deine Helferklassen
        base_distr = PerChannelDistribution(channel_distributions, self.channel_names)
        final_distr = DenormalizingDistribution(base_distr, stats)

        # Der alte `denorm`-Schritt am Ende entfällt, da dies jetzt im Wrapper passiert.
        # KORREKTUR: Gib die Wahrscheinlichkeitsmatrizen zurück, um die 8-Werte-Signatur zu erfüllen.
        return final_distr, base_distr, L_importance, avg_gate_weights_linear, avg_gate_weights_esn, expert_selection_counts, p_learned, p_final