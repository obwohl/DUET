import sys
import os
# Add project root to the Python path. This allows the script to be run directly,
# as it resolves the `ts_benchmark` module not found error.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

import unittest
import torch
from unittest.mock import patch
import traceback

# Die zu testende Klasse
from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import ReservoirExpert

# Helferklasse für die Konfiguration
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestLinearExtractorCluster(unittest.TestCase):

    def setUp(self):
        """Erstellt eine hybride Konfiguration für den Test."""
        self.config = dotdict({
            # MoE Konfiguration
            "num_linear_experts": 2,
            "num_esn_experts": 2,
            "k": 2,
            "noisy_gating": True,
            "loss_coef": 1.0,

            # Architektur-Parameter für Sub-Module
            "d_model": 32,
            "seq_len": 64,
            "input_size": 64,
            "hidden_size": 128,
            "enc_in": 3,
            "CI": False,
            "moving_avg": 25, # <<< FEHLENDER PARAMETER HINZUGEFÜGT

            # ESN-Experten Konfiguration
            "reservoir_size": 64,
            "spectral_radius": 0.99,
            "sparsity": 0.1,
            "input_scaling": 1.0,
        })

        # Wichtige Dimensionen
        self.B = 8  # Batch
        self.L = self.config.seq_len # Länge
        self.N = self.config.enc_in # Variablen

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"\n--- Running Cluster test on device: {self.device} ---")

    def test_01_integration_with_hybrid_experts(self):
        """
        Test 1: Funktioniert die Initialisierung, der Forward- und Backward-Pass
        mit einem gemischten Pool aus linearen und ESN-Experten?
        """
        try:
            cluster = Linear_extractor_cluster(self.config).to(self.device)
        except Exception as e:
            self.fail(f"Initialisierung des Clusters mit hybrider Konfiguration fehlgeschlagen: {e}\n{traceback.format_exc()}")

        # Überprüfe, ob die Gesamtanzahl der Experten korrekt ist
        total_experts = self.config.num_linear_experts + self.config.num_esn_experts
        self.assertEqual(len(cluster.experts), total_experts, "Die Anzahl der Experten im Cluster ist falsch.")

        # Erstelle Input-Daten
        input_tensor = torch.randn(self.B, self.L, self.N).to(self.device)

        # Teste Forward-Pass
        try:
            output, loss, _, _, selection_counts = cluster(input_tensor)
        except Exception as e:
            self.fail(f"Forward-Pass des Clusters fehlgeschlagen: {e}\n{traceback.format_exc()}")

        # Überprüfe Output-Formen
        self.assertEqual(output.shape, (self.B, self.config.d_model, self.N), "Output-Form des Clusters ist falsch.")
        self.assertEqual(loss.shape, torch.Size([]), "Loss-Form des Clusters ist falsch.")

        # Teste Backward-Pass (entscheidend!)
        # Simuliert das Training des Gating-Netzwerks
        total_loss = output.mean() + loss
        total_loss.backward()

        # Überprüfe, ob das Gating-Netzwerk einen Gradienten erhalten hat.
        # Das beweist, dass es von den Experten (egal welchen Typs) lernen kann.
        gate_param = next(cluster.gate.parameters())
        self.assertIsNotNone(gate_param.grad, "Das Gating-Netzwerk sollte einen Gradienten haben.")
        self.assertGreater(torch.abs(gate_param.grad).sum(), 0, "Die Summe der Gradienten des Gates sollte nicht Null sein.")

    def test_02_robustness_with_empty_gates(self):
        """
        Test 2 ('Gemeiner Test'): Was passiert, wenn das Gating keine Experten auswählt?
        Das kann passieren, wenn die Logits sehr negativ sind und nach top-k nichts übrig bleibt.
        """
        cluster = Linear_extractor_cluster(self.config).to(self.device)
        input_tensor = torch.randn(self.B, self.L, self.N).to(self.device)

        # Wir "patchen" die noisy_top_k_gating Methode, um leere Gates zu simulieren.
        # Der erste Rückgabewert (gates) ist ein Tensor voller Nullen.
        # Der zweite (load) ist ebenfalls ein Tensor voller Nullen.
        with patch.object(cluster, 'noisy_top_k_gating', return_value=(
            torch.zeros(self.B, self.config.num_linear_experts + self.config.num_esn_experts, device=self.device),
            torch.zeros(self.config.num_linear_experts + self.config.num_esn_experts, device=self.device)
             )):
            
            # Der Forward-Pass sollte auch mit leeren Gates nicht fehlschlagen
            output, loss, _, _, selection_counts = cluster(input_tensor)

            # Überprüfe, ob der Output ein korrekt geformter Null-Tensor ist
            self.assertEqual(output.shape, (self.B, self.config.d_model, self.N))
            self.assertTrue(torch.all(output == 0), "Output sollte bei leeren Gates ein Null-Tensor sein.")
            self.assertFalse(torch.isnan(loss).any(), "Loss sollte bei leeren Gates nicht NaN sein.")

    def test_03_robustness_with_only_chaotic_esn(self):
        """
        Test 3 ('Gemeinster Test'): Was passiert bei einer Konfiguration nur mit instabilen ESNs?
        Testet die numerische Stabilität des gesamten Forward-Passes unter Stress.
        """
        chaotic_config = self.config.copy()
        chaotic_config = dotdict(chaotic_config) # <<< NEU: Sicherstellen, dass es ein dotdict ist
        chaotic_config.num_linear_experts = 0
        chaotic_config.num_esn_experts = 1
        chaotic_config.k = 1 # Erzwinge die Auswahl des einzigen Experten
        chaotic_config.esn_configs = [
            {"spectral_radius": 1.5} # Sehr instabil
        ]

        cluster = Linear_extractor_cluster(dotdict(chaotic_config)).to(self.device)
        input_tensor = torch.randn(self.B, self.L, self.N).to(self.device)

        try:
            output, loss, _, _, selection_counts = cluster(input_tensor)
        except Exception as e:
            self.fail(f"Forward-Pass mit chaotischem ESN fehlgeschlagen: {e}")
        
        # Wir erwarten keinen Absturz, aber wir überprüfen auf NaNs/Infs
        self.assertFalse(torch.isnan(output).any(), "Output sollte bei chaotischem ESN nicht NaN sein.")
        self.assertFalse(torch.isinf(output).any(), "Output sollte bei chaotischem ESN nicht inf sein.")
        self.assertFalse(torch.isnan(loss).any(), "Loss sollte bei chaotischem ESN nicht NaN sein.")
        self.assertFalse(torch.isinf(loss).any(), "Loss sollte bei chaotischem ESN nicht inf sein.")

        # Überprüfe den Gradientenfluss auch in diesem Fall
        total_loss = output.mean() + loss
        total_loss.backward()
        gate_param = next(cluster.gate.parameters())
        self.assertIsNotNone(gate_param.grad, "Das Gating-Netzwerk sollte auch bei chaotischen ESNs einen Gradienten haben.")
        
        # Überprüfe, ob der ESN-Readout-Layer einen Gradienten hat
        esn_expert = [exp for exp in cluster.experts if isinstance(exp, ReservoirExpert)][0]
        self.assertIsNotNone(esn_expert.readout.weight.grad, "Der Readout-Layer des ESN sollte einen Gradienten haben.")

    def test_04_expert_selection_counts_constant_sum(self):
        """
        Test 4: Überprüft die Logik der `expert_selection_counts`.
        Die Summe der Zählungen über alle Experten muss pro Batch immer
        genau `batch_size * k` sein. Dies validiert das "Konstantsummenspiel".
        """
        cluster = Linear_extractor_cluster(self.config).to(self.device)
        input_tensor = torch.randn(self.B, self.L, self.N).to(self.device)

        # Führe den Forward-Pass aus, um die Zählungen zu erhalten
        _, _, _, _, selection_counts = cluster(input_tensor)

        # 1. Überprüfe die Form
        total_experts = self.config.num_linear_experts + self.config.num_esn_experts
        self.assertEqual(selection_counts.shape, (total_experts,), 
                         f"Die Form der selection_counts sollte [{total_experts}] sein.")

        # 2. Überprüfe die Summe (das ist der entscheidende Test)
        expected_sum = self.B * self.config.k  # batch_size * k
        actual_sum = torch.sum(selection_counts).item()

        self.assertEqual(actual_sum, expected_sum,
                         f"Die Summe der Expertenauswahlen muss {expected_sum} (batch_size * k) sein, war aber {actual_sum}.")

    def _run_weight_sum_test(self, config, test_name):
        """
        Hilfsfunktion, die die Konstantsummen-Eigenschaft der Gating-Gewichte für eine gegebene Konfiguration testet.
        """
        cluster = Linear_extractor_cluster(config).to(self.device)
        input_tensor = torch.randn(self.B, self.L, self.N).to(self.device)

        # Führe den Forward-Pass aus, um die durchschnittlichen Gewichte zu erhalten
        _, _, avg_weights_linear, avg_weights_esn, _ = cluster(input_tensor)

        # Summiere alle durchschnittlichen Gewichte
        total_avg_weight = torch.sum(avg_weights_linear) + torch.sum(avg_weights_esn)

        # Die Summe sollte sehr nahe an 1.0 liegen (aufgrund von Floating-Point-Arithmetik)
        self.assertAlmostEqual(total_avg_weight.item(), 1.0, places=5,
                               msg=f"Die Summe der durchschnittlichen Gating-Gewichte sollte 1.0 sein, war aber {total_avg_weight.item()} für den Test: '{test_name}'")

    def test_05_gating_weights_constant_sum(self):
        """
        Test 5: Überprüft, ob die Summe der durchschnittlichen Gating-Gewichte über alle Experten 1.0 ist.
        Dieser Test wird für verschiedene Experten-Konfigurationen ausgeführt.
        """
        # Konfiguration 1: Nur lineare Experten
        config_linear_only = dotdict(self.config.copy())
        config_linear_only.num_linear_experts = 4
        config_linear_only.num_esn_experts = 0
        config_linear_only.k = 2
        self._run_weight_sum_test(config_linear_only, "Nur lineare Experten")

        # Konfiguration 2: Nur ESN-Experten
        config_esn_only = dotdict(self.config.copy())
        config_esn_only.num_linear_experts = 0
        config_esn_only.num_esn_experts = 4
        config_esn_only.k = 2
        self._run_weight_sum_test(config_esn_only, "Nur ESN-Experten")

        # Konfiguration 3: Hybride Experten (aus dem setUp)
        self._run_weight_sum_test(self.config, "Hybride Experten (2+2)")

        # Konfiguration 4: Mehr Experten als k
        config_more_experts = dotdict(self.config.copy())
        config_more_experts.num_linear_experts = 5
        config_more_experts.num_esn_experts = 5
        config_more_experts.k = 4
        self._run_weight_sum_test(config_more_experts, "Mehr Experten als k (5+5, k=4)")


if __name__ == '__main__':
    unittest.main()