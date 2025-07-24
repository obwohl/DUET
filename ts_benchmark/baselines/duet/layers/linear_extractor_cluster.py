import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .distributional_router_encoder import encoder
from .expert_factory import create_experts 
from .esn.reservoir_expert import ReservoirExpert
from einops import rearrange
from .RevIN import RevIN  # Importiere RevIN


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        nonzero_locations = torch.nonzero(gates)
        if nonzero_locations.numel() == 0:
            self._expert_index = torch.tensor([], dtype=torch.long, device=gates.device)
            self._batch_index = torch.tensor([], dtype=torch.long, device=gates.device)
            self._part_sizes = [0] * num_experts
            self._nonzero_gates = torch.tensor([], device=gates.device)
            return

        sorted_experts, index_sorted_experts = nonzero_locations.sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = nonzero_locations[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        if self._batch_index.numel() == 0:
          # KORREKTUR: Gib leere Tensoren mit der passenden Dimensionalität zurück.
          # Der Input `inp` hat die Form [Batch * N_Vars, SeqLen, 1]
          # Wir wollen eine Liste von Tensoren der Form [0, SeqLen, 1] zurückgeben.
          dummy_shape = [0] + list(inp.shape[1:])
          return [torch.empty(*dummy_shape, device=inp.device, dtype=inp.dtype)
                  for _ in range(self._num_experts)]

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # Sicherstellen, dass die Batch-Größe aus _gates verwendet wird,
        # auch wenn expert_out leer ist oder leere Tensoren enthält.
        output_shape = [self._gates.size(0)] + list(expert_out[0].shape[1:]) if expert_out else [self._gates.size(0), 0, 0]
        combined = torch.zeros(*output_shape, device=self._gates.device)

        expert_out_filtered = [t for t in expert_out if t.numel() > 0]  # Filter non-empty
        if expert_out_filtered:
            stitched = torch.cat(expert_out_filtered, 0)
            if multiply_by_gates and self._nonzero_gates.numel() > 0:
                stitched = torch.einsum("i...,i...->i...", stitched, self._nonzero_gates)
            combined = combined.index_add(0, self._batch_index, stitched.float())

        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Linear_extractor_cluster(nn.Module):
    def __init__(self, config):
        super(Linear_extractor_cluster, self).__init__()
        self.noisy_gating = getattr(config, 'noisy_gating', True)
        
        self.num_experts = getattr(config, 'num_linear_experts', 0) + getattr(config, 'num_esn_experts', 0)

        self.input_size = config.seq_len 
        self.k = config.k

        self.experts = create_experts(config)

        # Füge die RevIN-Schicht hinzu
        self.revin = RevIN(
            num_features=config.enc_in,
            # Der norm_mode steuert die Art der Normalisierung.
            # 'subtract_median' ist das neue bevorzugte Verhalten.
            norm_mode=getattr(config, 'norm_mode', 'subtract_median')
        )

        self.gate = encoder(config, num_experts=self.num_experts)
        self.noise = encoder(config, num_experts=self.num_experts)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        safe_noise_stddev = torch.clamp(noise_stddev, min=1e-9)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / safe_noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / safe_noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        
        pre_softmax_logits_for_diag = None

        if False:
            print(f"\n--- GATING DIAGNOSTICS ---")
            print(f"clean_logits mean: {clean_logits.mean().item():.4f}, std: {clean_logits.std().item():.4f}")
            print(f"Sample clean_logits[0]: {clean_logits[0].detach().cpu().numpy()}")

        clean_logits = torch.nan_to_num(clean_logits)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon

            noise_stddev = torch.nan_to_num(noise_stddev, nan=noise_epsilon, posinf=5.0)

            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits
            if False:
                print(f"noisy_logits mean: {noisy_logits.mean().item():.4f}, std: {noisy_logits.std().item():.4f}")
        else:
            logits = clean_logits
        
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        
        pre_softmax_logits_for_diag = logits
        
        logits = self.softmax(logits)

        if False:
            print(f"Pre-Softmax logits[0]: {pre_softmax_logits_for_diag[0].detach().cpu().numpy()}")
            print(f"Probabilities (logits after softmax) mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
            print(f"Sample Probabilities[0]: {logits[0].detach().cpu().numpy()}")
            print(f"--------------------------\n")

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        gates, load = self.noisy_top_k_gating(x, self.training)
        
        importance = gates.sum(0)
        loss_importance = self.cv_squared(importance) + self.cv_squared(load)
        
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = []
        for i in range(self.num_experts):
            expert = self.experts[i]
            expert_input = expert_inputs[i]

            if not isinstance(expert, ReservoirExpert):
                expert_output = expert(expert_input)
                expert_outputs.append(expert_output)
                continue
            
            expert_input = rearrange(expert_input, 'b l n -> (b n) l')
            expert_output = expert(expert_input)
            expert_output = rearrange(expert_output, '(b n) d -> b d n', n=x.shape[-1])
            expert_outputs.append(expert_output)

        y = dispatcher.combine(expert_outputs)

        # Collect selection counts per expert type
        expert_selection_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
          expert_selection_counts[i] = torch.sum(gates[:, i] > 0)

        # --- KORRIGIERTE LOGIK zur Berechnung der durchschnittlichen Gating-Gewichte ---
        
        # Wir erstellen Listen, um die Durchschnittsgewichte für jeden Typ zu sammeln.
        linear_weights = []
        esn_weights = []

        for i in range(self.num_experts):
            # Berechne das Durchschnittsgewicht für diesen Experten über den gesamten Batch.
            avg_weight_for_expert_i = torch.mean(gates[:, i])
            
            # Ordne das Gewicht der richtigen Liste basierend auf dem Expertentyp zu.
            # WICHTIG: Der Klassenname muss exakt mit der Klassendefinition übereinstimmen.
            # Basierend auf den Logs aus `duet_prob.py` ist der Name 'Linear_extractor'.
            expert_class_name = self.experts[i].__class__.__name__
            
            if expert_class_name == 'Linear_extractor':
                linear_weights.append(avg_weight_for_expert_i)
            elif expert_class_name == 'ReservoirExpert':
                esn_weights.append(avg_weight_for_expert_i)

        # Konvertiere die Listen in Tensoren.
        # Dies behandelt den Fall, dass ein Expertentyp nicht vorhanden ist (leere Liste -> leerer Tensor).
        if linear_weights:
            avg_gate_weights_linear = torch.stack(linear_weights)
        else:
            avg_gate_weights_linear = torch.tensor([], device=x.device)

        if esn_weights:
            avg_gate_weights_esn = torch.stack(esn_weights)
        else:
            avg_gate_weights_esn = torch.tensor([], device=x.device)

        # Die Rückgabewerte sind jetzt saubere Tensoren, die nur die relevanten Gewichte enthalten.
        # Die nachgelagerte Logik in `duet_prob.py` verarbeitet diese korrekt.
        return y, loss_importance, avg_gate_weights_linear, avg_gate_weights_esn, expert_selection_counts