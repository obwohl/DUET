import unittest
import torch
import numpy as np

# Die Klasse, die wir testen werden (wird im nächsten Schritt erstellt)
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import ReservoirExpert

# Eine einfache Helferklasse, um ein Dictionary wie ein Objekt zu verwenden (für die config)
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestReservoirExpert(unittest.TestCase):

    def setUp(self):
        """Wird vor jedem Test aufgerufen, um eine Standardkonfiguration zu erstellen."""
        self.config = dotdict({
            "d_model": 32,
            "reservoir_size": 128,
            "spectral_radius": 0.99,
            "sparsity": 0.2,
            "input_scaling": 1.0,
        })

        self.B = 8  # Batch-Größe
        self.L = 64 # Sequenzlänge

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"\n--- Running ESN test on device: {self.device} ---")

    def test_01_initialization_shape_and_device(self):
        """Test 1: Werden alle internen Tensoren mit den korrekten Formen und auf dem richtigen Gerät erstellt?"""
        expert = ReservoirExpert(self.config).to(self.device)

        # Teste die Form der internen, nicht-trainierbaren Gewichte
        self.assertEqual(expert.W_in.shape, (self.config.reservoir_size, 1), "W_in shape is incorrect.")
        self.assertEqual(expert.W_res.shape, (self.config.reservoir_size, self.config.reservoir_size), "W_res shape is incorrect.")
        
        # Teste die Form der trainierbaren Ausgabeschicht
        self.assertEqual(expert.readout.in_features, self.config.reservoir_size, "Readout layer in_features is incorrect.")
        self.assertEqual(expert.readout.out_features, self.config.d_model, "Readout layer out_features is incorrect.")

        # Teste, ob alle auf dem richtigen Gerät sind
        self.assertEqual(expert.W_in.device.type, self.device.type, "W_in is on the wrong device.")
        self.assertEqual(expert.W_res.device.type, self.device.type, "W_res is on the wrong device.")
        self.assertEqual(next(expert.readout.parameters()).device.type, self.device.type, "Readout layer is on the wrong device.")

    def test_02_initialization_hyperparams(self):
        """Test 2: Werden die Hyperparameter (Sparsity, Spectral Radius) korrekt angewendet?"""
        expert = ReservoirExpert(self.config).to(self.device)
        
        # Teste Sparsity
        num_zeros = torch.sum(expert.W_res == 0).item()
        total_elements = expert.W_res.numel()
        actual_sparsity = num_zeros / total_elements
        self.assertAlmostEqual(actual_sparsity, self.config.sparsity, places=2, msg="Sparsity is not correctly applied.")

        # Teste Spectral Radius
        # Berechne die Eigenwerte der Reservoir-Matrix
        # MPS-FIX: Eigenwertberechnung muss auf der CPU erfolgen
        eigenvalues = torch.linalg.eigvals(expert.W_res.to('cpu'))
        # Der Spektralradius ist der maximale Absolutbetrag der Eigenwerte
        actual_spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        self.assertAlmostEqual(actual_spectral_radius, self.config.spectral_radius, places=4, msg="Spectral radius is not correctly set.")

    def test_03_forward_pass_shape(self):
        """Test 3: Liefert der Forward-Pass die korrekte Output-Form für volle und leere Batches?"""
        expert = ReservoirExpert(self.config).to(self.device)
        
        # Test mit einem normalen Batch
        input_tensor = torch.randn(self.B, self.L).to(self.device)
        output = expert(input_tensor)
        self.assertEqual(output.shape, (self.B, self.config.d_model), "Output shape for a full batch is incorrect.")

        # Test mit einem leeren Batch (wichtig für SparseDispatcher)
        empty_input = torch.tensor([]).reshape(0, self.L).to(self.device)
        empty_output = expert(empty_input)
        self.assertEqual(empty_output.shape, (0, self.config.d_model), "Output shape for an empty batch is incorrect.")

    def test_04_gradient_flow(self):
        """Test 4 ('Gemeiner Test'): Fließt der Gradient NUR durch die Ausgabeschicht?"""
        expert = ReservoirExpert(self.config).to(self.device)
        input_tensor = torch.randn(self.B, self.L).to(self.device)

        # Forward-Pass
        output = expert(input_tensor)
        
        # Loss berechnen und Backward-Pass
        loss = output.mean()
        loss.backward()

        # Überprüfe die Gradienten
        # Die Ausgabeschicht MUSS einen Gradienten haben
        self.assertIsNotNone(expert.readout.weight.grad, "Readout layer weight should have a gradient.")
        self.assertGreater(torch.abs(expert.readout.weight.grad).sum(), 0, "Readout layer weight gradient sum should be non-zero.")

        # Die Reservoir-Matrizen DÜRFEN KEINEN Gradienten haben
        self.assertIsNone(expert.W_res.grad, "W_res should not have a gradient.")
        self.assertIsNone(expert.W_in.grad, "W_in should not have a gradient.")

    def test_05_numerical_stability(self):
        """Test 5 ('Gemeinster Test'): Bleibt das Modell bei schwierigen Inputs numerisch stabil?"""
        # a) Stabiler Experte mit konstantem Input
        expert_stable = ReservoirExpert(self.config).to(self.device)
        constant_input = torch.ones(self.B, self.L).to(self.device) * 5.0
        output_stable = expert_stable(constant_input)
        self.assertFalse(torch.isnan(output_stable).any(), "Output should not be NaN for constant input (stable ESN).")
        self.assertFalse(torch.isinf(output_stable).any(), "Output should not be inf for constant input (stable ESN).")

        # b) Chaotischer Experte mit normalem Input
        # KORREKTUR: .copy() gibt ein dict zurück. Wir brauchen wieder ein dotdict.
        chaotic_config = dotdict(self.config.copy())
        chaotic_config.spectral_radius = 1.2 # Macht das Reservoir instabil/chaotisch
        expert_chaotic = ReservoirExpert(chaotic_config).to(self.device)
        random_input = torch.randn(self.B, self.L).to(self.device)
        output_chaotic = expert_chaotic(random_input)
        self.assertFalse(torch.isnan(output_chaotic).any(), "Output should not be NaN for normal input (chaotic ESN).")
        self.assertFalse(torch.isinf(output_chaotic).any(), "Output should not be inf for normal input (chaotic ESN).")

        # c) Stabiler Experte mit extremem Input
        extreme_input = torch.randn(self.B, self.L).to(self.device) * 1e6
        output_extreme = expert_stable(extreme_input)
        self.assertFalse(torch.isnan(output_extreme).any(), "Output should not be NaN for extreme input.")
        self.assertFalse(torch.isinf(output_extreme).any(), "Output should not be inf for extreme input.")


if __name__ == '__main__':
    unittest.main()