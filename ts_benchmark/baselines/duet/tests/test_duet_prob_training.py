import unittest
import torch
import pandas as pd
import numpy as np
from unittest import mock

# Importiere die zu testenden Klassen und Funktionen
from ..duet_prob import DUETProb, TransformerConfig
from ..models.duet_prob_model import DUETProbModel, DenormalizingDistribution, PerChannelDistribution
from ..utils.crps import crps_loss

class TestDUETProbTrainingAndModelLogic(unittest.TestCase):

    def setUp(self):
        """
        Diese Methode wird vor jedem einzelnen Test aufgerufen.
        """
        self.config_dict = {
            "seq_len": 24,
            "horizon": 12,
            "d_model": 16,
            "d_ff": 32,
            "n_heads": 2,
            "e_layers": 1,
            "enc_in": 2,
            "num_linear_experts": 1,
            "num_esn_experts": 1,
            "k": 1,
            "channel_bounds": {
                'temp': {'lower': -10, 'upper': 10},
                'pressure': {'lower': 900, 'upper': 1100},
            },
            "num_epochs": 1, # Nur für Tests
            "patience": 1,
            "batch_size": 4,
            "lr": 1e-4,
            "lradj": "constant",
            "loss_coef": 0.01,
            "num_workers": 0,
        }
        self.config = TransformerConfig(**self.config_dict)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def test_01_normalize_value_method(self):
        """
        Testet die `normalize_value` Methode der DenormalizingDistribution.
        """
        B, N_VARS = 2, 3
        
        # Erstelle bekannte Statistiken
        mean = torch.tensor([[10.0, 20.0, 1000.0]] * B) # Shape [B, N_VARS]
        std = torch.tensor([[2.0, 5.0, 50.0]] * B)     # Shape [B, N_VARS]
        stats = torch.stack([mean, std], dim=2)      # Shape [B, N_VARS, 2]

        # Erstelle eine Dummy-Basis-Verteilung (wird nicht direkt verwendet)
        dummy_base_dist = mock.Mock(spec=PerChannelDistribution)
        
        # Erstelle die denormalisierende Verteilung
        denorm_dist = DenormalizingDistribution(dummy_base_dist, stats)

        # Erstelle einen Wert auf der Original-Skala
        original_value = torch.randn(B, self.config.horizon, N_VARS) * std.unsqueeze(1) + mean.unsqueeze(1)
        
        # Normalisiere den Wert
        normalized_value = denorm_dist.normalize_value(original_value)

        # Erwartetes Ergebnis
        expected_normalized_value = (original_value - mean.unsqueeze(1)) / std.unsqueeze(1)

        self.assertTrue(torch.allclose(normalized_value, expected_normalized_value, atol=1e-6),
                        "normalize_value sollte die Skalierung korrekt umkehren.")

    def test_02_forward_pass_returns_correct_types_and_shapes(self):
        """
        Testet, ob der forward-Pass von DUETProbModel die korrekten Typen und Formen zurückgibt.
        """
        model = DUETProbModel(self.config).to(self.device)
        input_tensor = torch.randn(self.config.batch_size, self.config.seq_len, self.config.enc_in).to(self.device)

        denorm_distr, base_distr, loss_importance, _, _, _, _ = model(input_tensor)

        # Teste Typen
        self.assertIsInstance(denorm_distr, DenormalizingDistribution)
        self.assertIsInstance(base_distr, PerChannelDistribution)
        self.assertIsInstance(loss_importance, torch.Tensor)

        # Teste Shapes (indirekt über die Methoden)
        # Denormalisierte Vorhersage
        denorm_preds = denorm_distr.icdf(0.5)
        self.assertEqual(denorm_preds.shape, (self.config.batch_size, self.config.horizon, self.config.enc_in))
        
        # Normalisierte Vorhersage
        norm_preds = base_distr.icdf(0.5)
        self.assertEqual(norm_preds.shape, (self.config.batch_size, self.config.horizon, self.config.enc_in))

    def test_03_normalized_vs_denormalized_loss_values(self):
        """
        Stellt sicher, dass der normalisierte und denormalisierte Loss unterschiedliche Werte haben,
        insbesondere bei Kanälen mit unterschiedlichen Skalen.
        """
        model = DUETProbModel(self.config).to(self.device)
        # --- KORREKTUR: Der Input muss die gleiche Skalierung wie der Output haben,
        # damit RevIN korrekte Statistiken für die Denormalisierung berechnet.
        input_temp = torch.randn(self.config.batch_size, self.config.seq_len, 1) # Skala um 0
        input_pressure = torch.randn(self.config.batch_size, self.config.seq_len, 1) * 50 + 1000 # Skala um 1000
        input_tensor = torch.cat([input_temp, input_pressure], dim=2).to(self.device)
        
        # Erstelle Zielwerte mit unterschiedlichen Skalen
        target_temp = torch.randn(self.config.batch_size, self.config.horizon, 1) # Skala um 0
        target_pressure = torch.randn(self.config.batch_size, self.config.horizon, 1) * 50 + 1000 # Skala um 1000
        target_horizon = torch.cat([target_temp, target_pressure], dim=2).to(self.device)

        # Führe Forward-Pass aus
        denorm_distr, base_distr, _, _, _, _, _ = model(input_tensor)

        # Berechne den denormalisierten Loss
        denorm_loss = crps_loss(denorm_distr, target_horizon.permute(0, 2, 1)).mean()

        # Berechne den normalisierten Loss
        norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
        norm_loss = crps_loss(base_distr, norm_target).mean()

        # Der denormalisierte Loss sollte aufgrund des Druckkanals viel größer sein
        self.assertGreater(denorm_loss.item(), norm_loss.item() * 10, 
                           "Der denormalisierte Loss sollte aufgrund des Skalierungsunterschieds signifikant größer sein als der normalisierte.")
        
    def test_04_gradient_flow_with_normalized_loss(self):
        """
        Testet, ob Gradienten korrekt fließen, wenn der Loss auf der normalisierten
        Verteilung berechnet wird.
        """
        model = DUETProbModel(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        input_tensor = torch.randn(self.config.batch_size, self.config.seq_len, self.config.enc_in).to(self.device)
        target_horizon = torch.randn(self.config.batch_size, self.config.horizon, self.config.enc_in).to(self.device)

        optimizer.zero_grad()

        # Forward-Pass
        denorm_distr, base_distr, loss_importance, _, _, _, _ = model(input_tensor)

        # Berechne den normalisierten Loss (wie im Training)
        norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
        normalized_crps_loss = crps_loss(base_distr, norm_target).mean()
        total_loss = normalized_crps_loss + self.config.loss_coef * loss_importance

        # Backward-Pass
        total_loss.backward()

        # Überprüfe, ob ein wichtiger Parameter einen Gradienten hat
        # --- KORREKTUR: Der Zugriff auf die Gewichte muss die neue MLPProjectionHead-Struktur berücksichtigen ---
        if model.args_proj.num_layers > 0:
            # Wenn es Residual-Layer gibt, prüfen wir die finale Schicht
            param_with_grad = model.args_proj.final_layer.weight
        else:
            # Ansonsten prüfen wir die einzelne lineare Fallback-Schicht
            param_with_grad = model.args_proj.projection.weight

        self.assertIsNotNone(param_with_grad.grad, "Gradienten sollten nach dem backward-Pass nicht None sein.")
        self.assertGreater(torch.abs(param_with_grad.grad).sum(), 0, "Die Summe der Gradienten sollte nicht Null sein.")

    @mock.patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    @mock.patch.object(DUETProb, 'load') # Mocke die load-Methode, um den FileNotFoundError zu verhindern
    @mock.patch('torch.save')
    def test_05_full_training_loop_with_mlp_head(self, mock_torch_save, mock_load, mock_summary_writer):
        """
        Testet, ob der gesamte Trainings- und Validierungszyklus (forecast_fit)
        erfolgreich mit einem aktivierten MLP Projection Head durchläuft.
        """
        print("\nRunning test: Full training loop with MLP Head...")
        # Erstelle eine Konfiguration, die den MLP-Kopf explizit aktiviert
        mlp_head_config_dict = self.config_dict.copy()
        mlp_head_config_dict['projection_head_layers'] = 1
        mlp_head_config_dict['projection_head_dim_factor'] = 4
        mlp_head_config_dict['projection_head_dropout'] = 0.2

        # Erstelle die Wrapper-Instanz mit der neuen Konfiguration
        wrapper = DUETProb(**mlp_head_config_dict)
        
        # Erstelle Dummy-Trainingsdaten
        train_len = 100
        dates = pd.date_range(start="2023-01-01", periods=train_len, freq="h")
        data = np.random.randn(train_len, self.config.enc_in)
        train_valid_data = pd.DataFrame(data, index=dates, columns=self.config.channel_bounds.keys())

        try:
            # Führe die Haupt-Trainingsmethode aus. Ein erfolgreicher Durchlauf ohne Fehler ist der Test.
            wrapper.forecast_fit(train_valid_data, train_ratio_in_tv=0.8)
            print("OK")
        except Exception as e:
            import traceback
            self.fail(f"forecast_fit failed with an MLP projection head. Error: {e}\n{traceback.format_exc()}")