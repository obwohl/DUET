import unittest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Fügen Sie den Projekt-Stammordner zum Pfad hinzu, um Importe zu ermöglichen
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.duet_prob import DUETProb

class TestDUETProbPlottingLogic(unittest.TestCase):

    def setUp(self):
        """Bereitet eine minimale Instanz von DUETProb für die Tests vor."""
        self.seq_len = 96
        self.horizon = 24
        
        self.wrapper = DUETProb(seq_len=self.seq_len, horizon=self.horizon)
        
        # Mocken des Modells und des Datasets, da wir nur die Plot-Logik testen
        self.wrapper.model = MagicMock(spec=torch.nn.Module)
        # Stellen Sie sicher, dass das gemockte Modell Parameter hat, um 'next(self.model.parameters())' zu ermöglichen
        self.wrapper.model.parameters.return_value = iter([torch.nn.Parameter(torch.empty(1))])
        
        # --- KORREKTUR: Der Mock für `icdf` muss dynamisch auf die Anzahl der
        # angeforderten Quantile reagieren. `crps_loss` fordert 99 Quantile an,
        # während der Plot-Code die in der Konfiguration definierten Quantile verwendet.
        def mock_icdf_side_effect(quantiles_tensor):
            """Erstellt einen Tensor mit der korrekten Quantil-Dimension."""
            num_quantiles = len(quantiles_tensor) if quantiles_tensor.dim() > 0 else 1
            # Form: [Batch, Horizont, N_Vars, N_Quantiles]
            return torch.randn(1, self.horizon, 1, num_quantiles)

        # Mock für die Verteilungsobjekte mit Rückgabewerten, die sich wie Tensoren verhalten
        mock_dist = MagicMock()
        mock_dist.mean = torch.tensor(0.0)  # Simuliert ein Attribut, das oft verwendet wird
        # Verwende side_effect, um dynamisch auf die Eingabe zu reagieren.
        mock_dist.icdf.side_effect = mock_icdf_side_effect
        
        # Konfiguriere den Rückgabewert für den Aufruf des Modells
        self.wrapper.model.return_value = (
            mock_dist,          # denorm_distr
            mock_dist,          # base_distr
            torch.tensor(0.0),  # loss_importance
            MagicMock(),        # batch_gate_weights_linear
            MagicMock(),        # batch_gate_weights_esn
            MagicMock(),        # batch_selection_counts
            MagicMock()         # anchor_correction
        )

        # ... (Rest der setUp Methode bleibt unverändert)

        # Erstellen eines Dummy-Validierungsdatasets
        self.valid_data_len = 500
        self.mock_valid_dataset = MagicMock()
        self.mock_valid_dataset.__len__.return_value = self.valid_data_len
        
        # Konfigurieren der __getitem__-Methode des Mocks, um Dummy-Tensoren zurückzugeben
        dummy_input = torch.randn(self.seq_len, 1)
        dummy_target = torch.randn(self.seq_len + self.horizon, 1)
        self.mock_valid_dataset.__getitem__.return_value = (dummy_input, dummy_target, None, None)

    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._create_window_plot')
    def test_plot_logic_for_valid_index(self, mock_create_plot):
        """
        Testet, ob für einen gültigen Fenster-Index die Plot-Funktion aufgerufen wird.
        Ein "gültiger" Index ist einer, der weit genug vom Anfang entfernt ist.
        """
        # Ein "guter" Index, gefunden von find_interesting_windows
        valid_start_idx = 200
        self.wrapper.interesting_window_indices = {'channel_A': {'ks_dist': valid_start_idx}}

        # Führe die zu testende Funktion aus
        self.wrapper._log_interesting_window_plots(epoch=1, writer=MagicMock(), valid_dataset=self.mock_valid_dataset)

        # Überprüfung: Wurde die Plot-Funktion genau einmal aufgerufen?
        mock_create_plot.assert_called_once()

    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._create_window_plot')
    def test_plot_logic_skips_invalid_index_at_start(self, mock_create_plot):
        """
        Dies ist der entscheidende Test: Er simuliert das alte, fehlerhafte Verhalten.
        Wenn ein Index zu nah am Anfang liegt (keine ausreichende Historie),
        darf die Plot-Funktion NICHT aufgerufen werden.
        """
        # Ein "schlechter" Index, der von argmax([0,0,...]) zurückgegeben wurde
        # Dieser Index würde einen sample_idx < 0 erfordern.
        invalid_start_idx = 10
        self.wrapper.interesting_window_indices = {'channel_B': {'ks_dist': invalid_start_idx}}

        # Führe die zu testende Funktion aus
        self.wrapper._log_interesting_window_plots(epoch=1, writer=MagicMock(), valid_dataset=self.mock_valid_dataset)

        # Überprüfung: Wurde die Plot-Funktion NICHT aufgerufen?
        mock_create_plot.assert_not_called()

    def test_index_calculation_is_correct(self):
        """
        Testet die korrekte Berechnung des `sample_idx` aus dem `window_start_idx`.
        Dies ist die Kernlogik, die über Plot oder Skip entscheidet.
        """
        # Ein Fenster, das bei Index 250 im rohen Validierungsdatensatz beginnt.
        window_start_idx = 250
        
        # Das "Nachher"-Fenster, das wir vorhersagen wollen, beginnt bei:
        # window_start_idx + horizon
        forecast_start_idx = window_start_idx + self.horizon # 250 + 24 = 274
        
        # Der Input für diese Vorhersage ist das Fenster der Länge `seq_len`,
        # das bei `forecast_start_idx` endet. Der `forecasting_data_provider`
        # erstellt Samples, wobei das Sample `j` dem Input `raw_data[j : j + seq_len]` entspricht.
        # Daher ist der Index des Samples, das wir benötigen:
        # forecast_start_idx - seq_len
        expected_sample_idx = forecast_start_idx - self.seq_len # 274 - 96 = 178

        # Wir simulieren den Aufruf innerhalb der Funktion mit einem Patch,
        # um den berechneten Index abzufangen.
        with patch.object(self.mock_valid_dataset, '__getitem__') as mock_getitem:
            # Konfiguriere den Rückgabewert für den neuen Mock, erstellt durch den Patch.
            dummy_input = torch.randn(self.seq_len, 1)
            dummy_target = torch.randn(self.seq_len + self.horizon, 1)
            mock_getitem.return_value = (dummy_input, dummy_target, None, None)

            self.wrapper.interesting_window_indices = {'channel_C': {'some_method': window_start_idx}}
            self.wrapper._log_interesting_window_plots(epoch=1, writer=MagicMock(), valid_dataset=self.mock_valid_dataset)
            
            # Überprüfen, ob __getitem__ mit dem korrekt berechneten Index aufgerufen wurde.
            mock_getitem.assert_called_once_with(expected_sample_idx)

    def test_channel_specific_crps_in_plot_title(self):
        """
        Testet, ob der im Plot-Titel angezeigte CRPS-Wert kanalspezifisch ist
        und nicht ein globaler Durchschnitt, wie vom Benutzer gemeldet.
        """
        # 1. Konfiguration mit zwei Kanälen
        n_vars = 2
        self.wrapper.config.channel_bounds = {'channel_A': {}, 'channel_B': {}}

        # 2. Passe die Mocks an, um 2 Kanäle zu simulieren
        # a) Das Dataset gibt Daten mit 2 Kanälen zurück
        dummy_input = torch.randn(self.seq_len, n_vars)
        dummy_target = torch.randn(self.seq_len + self.horizon, n_vars)
        self.mock_valid_dataset.__getitem__.return_value = (dummy_input, dummy_target, None, None)

        # b) Die Verteilung (vom Modell) gibt Vorhersagen für 2 Kanäle zurück
        def mock_icdf_side_effect_2_vars(quantiles_tensor):
            num_quantiles = len(quantiles_tensor) if quantiles_tensor.dim() > 0 else 1
            return torch.randn(1, self.horizon, n_vars, num_quantiles)

        mock_dist = MagicMock()
        mock_dist.mean = torch.tensor(0.0)
        mock_dist.icdf.side_effect = mock_icdf_side_effect_2_vars
        self.wrapper.model.return_value = (mock_dist, mock_dist, torch.tensor(0.0), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # 3. Mock für crps_loss, der kanalspezifische Verluste zurückgibt
        # Shape: [B, N_vars, H] -> [1, 2, 24]
        # Kanal A (Index 0) hat einen niedrigen Loss, Kanal B (Index 1) einen hohen.
        crps_channel_A = torch.full((1, 1, self.horizon), 10.0)
        crps_channel_B = torch.full((1, 1, self.horizon), 500.0)
        mock_crps_return = torch.cat([crps_channel_A, crps_channel_B], dim=1)

        # 4. Setze interessante Fenster für beide Kanäle
        self.wrapper.interesting_window_indices = {
            'channel_A': {'method1': 200},
            'channel_B': {'method2': 210}
        }

        # 5. Patche die Plot-Funktion und die crps_loss-Funktion
        with patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._create_window_plot') as mock_create_plot, \
             patch('ts_benchmark.baselines.duet.duet_prob.crps_loss', return_value=mock_crps_return):

            self.wrapper._log_interesting_window_plots(epoch=1, writer=MagicMock(), valid_dataset=self.mock_valid_dataset)

            # Überprüfe die Aufrufe
            self.assertEqual(mock_create_plot.call_count, 2, "Die Plot-Funktion sollte für jeden Kanal einmal aufgerufen werden.")

            # Hole die 'title' Argumente aus den Aufrufen
            call_args_list = mock_create_plot.call_args_list
            titles = [call.kwargs['title'] for call in call_args_list]

            # Finde die Titel, die mit dem Kanalnamen beginnen
            title_A = next((t for t in titles if t.startswith('channel_A')), None)
            title_B = next((t for t in titles if t.startswith('channel_B')), None)

            self.assertIsNotNone(title_A, "Ein Titel für channel_A sollte vorhanden sein.")
            self.assertIsNotNone(title_B, "Ein Titel für channel_B sollte vorhanden sein.")

            # Überprüfe, ob die korrekten CRPS-Werte in den Titeln sind
            self.assertIn("CRPS: 10.0000", title_A, "Der CRPS-Wert für Kanal A ist im Titel falsch.")
            self.assertIn("CRPS: 500.0000", title_B, "Der CRPS-Wert für Kanal B ist im Titel falsch.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)