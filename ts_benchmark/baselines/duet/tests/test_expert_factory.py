import unittest
import torch
import torch.nn as nn

# Die zu testende Fabrik-Funktion
from ts_benchmark.baselines.duet.layers.expert_factory import create_experts

# Die Experten-Klassen, die die Fabrik erstellen soll
from ts_benchmark.baselines.duet.layers.linear_pattern_extractor import Linear_extractor
from ts_benchmark.baselines.duet.layers.esn.reservoir_expert import ReservoirExpert

# Helferklasse für die Konfiguration
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestExpertFactory(unittest.TestCase):

    def setUp(self):
        """Erstellt eine Standardkonfiguration für die Tests."""
        self.base_config = {
            "d_model": 32,
            "seq_len": 64,
            "input_size": 64,
            "hidden_size": 128,
            # Standard-ESN-Parameter (werden ggf. überschrieben)
            "reservoir_size": 128,
            "spectral_radius": 0.99,
            "sparsity": 0.1,
            "input_scaling": 1.0,
        }

    def test_01_creates_correct_number_and_type_of_experts(self):
        """Test 1: Erstellt die Fabrik die korrekte Anzahl und die richtigen Typen von Experten?"""
        config = dotdict(self.base_config.copy())
        config.num_linear_experts = 3
        config.num_esn_experts = 2
        
        experts_list = create_experts(config)

        # Überprüfe die Gesamtanzahl
        self.assertIsInstance(experts_list, nn.ModuleList, "Die Fabrik sollte eine nn.ModuleList zurückgeben.")
        self.assertEqual(len(experts_list), 5, "Die Gesamtanzahl der Experten ist falsch.")

        # Zähle die Typen
        num_linear = sum(isinstance(exp, Linear_extractor) for exp in experts_list)
        num_esn = sum(isinstance(exp, ReservoirExpert) for exp in experts_list)
        
        self.assertEqual(num_linear, 3, "Die Anzahl der linearen Experten ist falsch.")
        self.assertEqual(num_esn, 2, "Die Anzahl der ESN-Experten ist falsch.")

    def test_02_handles_zero_experts_of_one_type(self):
        """Test 2: Funktioniert die Fabrik auch, wenn ein Expertentyp nicht vorhanden ist?"""
        config = dotdict(self.base_config.copy())
        config.num_linear_experts = 5
        config.num_esn_experts = 0 # Kein ESN-Experte
        
        experts_list = create_experts(config)
        
        self.assertEqual(len(experts_list), 5)
        self.assertTrue(all(isinstance(exp, Linear_extractor) for exp in experts_list))

    def test_03_applies_diverse_esn_configs(self):
        """Test 3: Wendet die Fabrik die spezifischen Konfigurationen auf die ESN-Experten an?"""
        config = dotdict(self.base_config.copy())
        config.num_linear_experts = 1
        config.num_esn_experts = 2 # Wir wollen 2 ESN-Experten
        
        # Definiere spezifische Konfigurationen für die beiden ESN-Experten
        config.esn_configs = [
            {"spectral_radius": 0.9, "sparsity": 0.1},
            {"spectral_radius": 1.1, "sparsity": 0.3}
        ]

        experts_list = create_experts(config)
        
        # Extrahiere die ESN-Experten aus der Liste
        esn_experts = [exp for exp in experts_list if isinstance(exp, ReservoirExpert)]
        self.assertEqual(len(esn_experts), 2)

        # Überprüfe, ob die spezifischen Hyperparameter korrekt gesetzt wurden
        self.assertAlmostEqual(esn_experts[0].spectral_radius_target, 0.9)
        self.assertAlmostEqual(esn_experts[0].sparsity, 0.1)
        
        self.assertAlmostEqual(esn_experts[1].spectral_radius_target, 1.1)
        self.assertAlmostEqual(esn_experts[1].sparsity, 0.3)
        
        # Überprüfe, ob ein gemeinsamer Parameter korrekt gesetzt wurde
        self.assertEqual(esn_experts[0].reservoir_size, 128)
        self.assertEqual(esn_experts[1].reservoir_size, 128)

if __name__ == '__main__':
    unittest.main()

