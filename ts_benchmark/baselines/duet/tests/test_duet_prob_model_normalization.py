import unittest
import torch
import sys
import os

# Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestDUETProbModelNormalization(unittest.TestCase):

    def _create_config(self, channel_bounds: dict) -> dotdict:
        """Erstellt eine Konfiguration für den Test."""
        n_vars = len(channel_bounds)
        config = dotdict({
            "horizon": 16, "d_model": 32, "d_ff": 64, "n_heads": 4, "e_layers": 1,
            "enc_in": n_vars, "moving_avg": 25, "dropout": 0.1, "fc_dropout": 0.1,
            "factor": 3, "activation": "gelu", "output_attention": False, "CI": False,
            "num_linear_experts": 1, "num_esn_experts": 1, "hidden_size": 128,
            "k": 1, "noisy_gating": True, "reservoir_size": 64, "spectral_radius": 0.99,
            "sparsity": 0.1, "input_scaling": 1.0, "num_bins": 50, "tail_percentile": 0.05,
            "channel_bounds": channel_bounds, "seq_len": 64,
        })
        config.input_size = config.seq_len
        config.label_len = config.seq_len // 2
        config.pred_len = config.horizon
        return config

    def setUp(self):
        """Bereitet Geräte und Konfigurationen vor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 4
        self.L = 64

        # Konfiguration für Daten mit kleiner Magnitude (z.B. Temperatur)
        self.low_mag_bounds = {'temp': {'lower': -10, 'upper': 40}}
        self.config_low_mag = self._create_config(self.low_mag_bounds)

        # Konfiguration für Daten mit großer Magnitude (z.B. Luftdruck)
        self.high_mag_bounds = {'pressure': {'lower': 950, 'upper': 1050}}
        self.config_high_mag = self._create_config(self.high_mag_bounds)

    def test_normalization_bug_with_high_magnitude_data(self):
        """
        Dieser Test verifiziert den Fix für den Normalisierungs-Bug.

        Hypothese (vor dem Fix): Bei Daten mit hohem Mittelwert (z.B. 1000) führt
        die doppelte Denormalisierung dazu, dass die Vorhersage massiv von den
        Eingabewerten abweicht.

        Test-Logik:
        1. Erstelle ein Modell mit einer Konfiguration für Hoch-Magnitude-Daten.
        2. Erzeuge zufällige Eingabedaten und addiere einen großen Offset (1000).
        3. Führe den Forward-Pass aus, um eine denormalisierte Vorhersage zu erhalten.
        4. Überprüfe, ob der Mittelwert der Vorhersage in der gleichen Größenordnung
           liegt wie der Mittelwert der Eingabedaten. Ein großer Unterschied
           deutet auf den Bug hin.
        """
        print(f"\n--- Running Normalization Bug Verification Test on {self.device} ---")
        
        # 1. Modell mit Hoch-Magnitude-Konfiguration
        model = DUETProbModel(self.config_high_mag).to(self.device)
        model.eval()

        # 2. Erzeuge Hoch-Magnitude-Input
        # Zufällige Daten um 0, dann einen großen Offset hinzufügen
        input_data = torch.randn(self.B, self.L, self.config_high_mag.enc_in).to(self.device)
        offset = 1000.0
        input_data += offset

        # 3. Führe den Forward-Pass aus
        with torch.no_grad():
            # Unpack, um die finale, denormalisierte Verteilung zu erhalten
            final_dist, _, _, _, _, _ = model(input_data)
            
            # Hole die Median-Vorhersage (q=0.5) aus der Verteilung
            # .icdf gibt [B, H, N_vars] zurück
            median_prediction = final_dist.icdf(0.5)

        # 4. Überprüfe die Größenordnung der Vorhersage
        input_mean = input_data.mean().item()
        prediction_mean = median_prediction.mean().item()

        print(f"Input Mean: {input_mean:.2f}")
        print(f"Prediction Mean: {prediction_mean:.2f}")

        # Der Mittelwert der Vorhersage sollte nahe am Mittelwert des Inputs liegen.
        # Ein großer Unterschied (z.B. Faktor 2) deutet auf den Bug hin.
        # Wir erlauben eine großzügige Toleranz, da das Modell untrainiert ist,
        # aber die Skalierung muss stimmen.
        self.assertAlmostEqual(input_mean, prediction_mean, delta=input_mean * 0.5,
                               msg="Der Mittelwert der Vorhersage weicht massiv vom Mittelwert des Inputs ab. Der Normalisierungs-Bug ist wahrscheinlich noch vorhanden.")
        print("--- Test Passed: Prediction magnitude is correct. ---")

if __name__ == '__main__':
    unittest.main()