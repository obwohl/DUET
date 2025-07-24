import unittest
import torch
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
import shutil

# Add the project directory to the Python path to avoid import issues.
# This might need adjustment depending on your project structure.
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    from ts_benchmark.baselines.duet.duet_prob import DUETProb
    from ts_benchmark.baselines.duet.models.duet_prob_model import PerChannelDistribution
except (ImportError, ModuleNotFoundError):
    # Fallback if the structure is different or script is run from another location
    print("Could not import DUETProb from expected path. Running with local class definition.")
    DUETProb = DUETProb # Use the class defined in this file

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

class TestDUETProbWrapper(unittest.TestCase):

    def setUp(self):
        """Called before every test."""
        self.test_dir = "temp_test_runs"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # A minimal config for a small, real model
        self.params = {
            "seq_len": 4,
            "horizon": 8,
            "d_model": 8,
            "d_ff": 8,
            "n_heads": 1,
            "e_layers": 1,
            "num_linear_experts": 2,
            "num_esn_experts": 2,
            "k": 2,
            "hidden_size": 8,
            "num_epochs": 1,
            "batch_size": 4,
            "patience": 3,
            "lr": 1e-4,
            "loss_coef": 0.01,
            "CI": False,
            "quantiles": [0.1, 0.5, 0.9],
            "reservoir_size": 8,
            "spectral_radius": 0.98,
            "sparsity": 0.2,
            "input_scaling": 1.0,
            "loss_target_clip": None, # FIX: Add default for new param
            "norm_mode": "subtract_last", # Explicitly set norm_mode for tests
        }
        
        self.n_vars = 2
        self.train_len = 200
        dates = pd.date_range(start="2025-01-01", periods=self.train_len, freq="h")
        data = np.random.randn(self.train_len, self.n_vars)
        self.train_valid_data = pd.DataFrame(data, index=dates, columns=[f'var_{i}' for i in range(self.n_vars)])

    def tearDown(self):
        """Called after every test to clean up."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    @patch.object(DUETProb, 'load')
    @patch('torch.save')
    @patch.object(DUETProb, 'validate')
    def test_06_expert_metric_logging(self, mock_validate, mock_torch_save, mock_load, mock_SummaryWriter):
        """
        Test 6: Are expert metrics (gating weights, selection) correctly 
        averaged over all batches of an epoch and logged?
        """
        # FIX: The mock for validate must now return a tuple of two floats
        # to match the new signature in duet_prob.py
        mock_validate.return_value = (0.5, 0.5)
        mock_load.return_value = MagicMock()
        
        mock_writer_instance = mock_SummaryWriter.return_value
        
        # This length is crafted to create a valid training and validation set
        required_raw_len = 60
        small_train_data = self.train_valid_data.iloc[:required_raw_len]

        # Initialize with a minimal but real model configuration
        model_wrapper = DUETProb(**self.params)
        model_wrapper.config.log_dir = self.test_dir
        
        # Run forecast_fit with the real model within our mocked environment
        model_wrapper.forecast_fit(small_train_data, train_ratio_in_tv=0.8)

        # Check that the writer was called with the expert metric keys
        logged_keys = {call.args[0] for call in mock_writer_instance.add_scalar.call_args_list}
        
        # We don't know the exact values, but we can assert the keys were logged
        self.assertIn('Expert_Gating_Weights/Linear_Expert_0', logged_keys)
        self.assertIn('Expert_Gating_Weights/Reservoir_Expert_0', logged_keys)
        self.assertIn('Expert_Selection_Counts/Linear_extractor_0', logged_keys)
        self.assertIn('Expert_Selection_Counts/ReservoirExpert_0', logged_keys)

    def test_07_create_window_plot_generates_correct_figure(self):
        """
        Test 7: Stellt sicher, dass die interne Plot-Funktion `_create_window_plot`
        eine korrekte Matplotlib-Figur mit den erwarteten Elementen erzeugt.
        """
        # 1. Initialisiere den Wrapper mit einer minimalen Konfiguration für den Plot
        plot_params = self.params.copy()
        plot_params['seq_len'] = 16  # FIX: History must be >= horizon for plot logic
        plot_params['horizon'] = 8   # FIX: Set a valid horizon
        plot_params['quantiles'] = [0.1, 0.5, 0.9] # 1 Median, 1 CI-Level
        plot_params['channel_bounds'] = {'var_0': {}, 'var_1': {}} # Namen sind wichtig
        model_wrapper = DUETProb(**plot_params)

        # 2. Erstelle Dummy-Daten für den Plot
        history_data = np.random.randn(plot_params['seq_len'], self.n_vars)
        actuals_data = np.random.randn(plot_params['horizon'], self.n_vars)
        
        # 3. Erstelle eine Mock-Verteilung, die das Verhalten der echten nachahmt
        mock_prediction_dist = MagicMock()
        
        # Das `icdf`-Ergebnis muss die Form [B, H, V, Q] haben
        mock_quantile_preds = torch.randn(1, plot_params['horizon'], self.n_vars, len(plot_params['quantiles']))
        mock_prediction_dist.icdf.return_value = mock_quantile_preds
        
        # Die Methode benötigt ein Gerät von der Verteilung
        # Wir mocken das, indem wir es an ein beliebiges Tensor-Attribut anhängen
        mock_prediction_dist.mean = torch.tensor([], device='cpu')

        # 4. Rufe die zu testende Methode auf
        test_title = "Mein Test-Titel"
        fig = model_wrapper._create_window_plot(
            history=history_data,
            actuals=actuals_data,
            prediction_dist=mock_prediction_dist,
            channel_name='var_0',
            title=test_title
        )

        # 5. Überprüfe das Ergebnis
        self.assertIsInstance(fig, plt.Figure, "Die Methode sollte eine Matplotlib-Figur zurückgeben.")

        # Überprüfe den Titel
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), test_title, "Der Titel des Plots wurde nicht korrekt gesetzt.")

        # Überprüfe die gezeichneten Elemente
        # Wir erwarten 3 Linien: History, Actual, Median Forecast
        lines = ax.get_lines()
        self.assertEqual(len(lines), 3, "Es sollten genau 3 Linien (History, Actual, Median) gezeichnet werden.")

        # Wir erwarten 1 PolyCollection für das eine Konfidenzintervall (0.1-0.9)
        # ax.fill_between() creates a PolyCollection, not a single Polygon.
        poly_collections = [p for p in ax.get_children() if isinstance(p, PolyCollection)]
        self.assertEqual(len(poly_collections), 1, "Es sollte genau ein Konfidenzintervall (PolyCollection) gezeichnet werden.")
        
        # Schließe die Figur, um Speicherlecks in Tests zu vermeiden
        plt.close(fig)

    @patch('ts_benchmark.baselines.duet.duet_prob.crps_loss')
    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    @patch.object(DUETProb, 'load')
    @patch('torch.save')
    @patch.object(DUETProb, 'validate')
    def test_08_loss_target_clip_is_applied(self, mock_validate, mock_torch_save, mock_load, mock_SummaryWriter, mock_crps_loss):
        """
        Tests if the `loss_target_clip` hyperparameter correctly clamps the
        normalized target values before they are passed to the loss function.
        This is the CRPS-equivalent of the "learn from tails" trick.
        """
        # 1. Setup mocks
        mock_validate.return_value = (0.5, 0.5)

        # 2. Create data with a massive outlier
        seq_len, horizon, batch_size, n_vars = 4, 8, 4, 2
        # FIX: The mock must return a tensor with the correct shape for the .mean(dim=(...)) call
        mock_crps_loss.return_value = torch.randn(batch_size, n_vars, horizon, requires_grad=True)

        train_len = seq_len + horizon + batch_size 
        dates = pd.date_range(start="2025-01-01", periods=train_len, freq="h")
        data = np.random.randn(train_len, n_vars)
        spike_index = seq_len + 2
        data[spike_index, 0] = 1000.0 
        train_data = pd.DataFrame(data, index=dates, columns=[f'var_{i}' for i in range(n_vars)])

        # 3. Configure the model with clipping enabled
        clip_value = 5.0
        test_params = self.params.copy()
        test_params.update({
            'loss_target_clip': clip_value,
            'norm_mode': 'subtract_last', # This normalization is key to creating a large target value
            'num_epochs': 1,
            'seq_len': seq_len, 'horizon': horizon, 'batch_size': batch_size
        })
        
        model_wrapper = DUETProb(**test_params)
        model_wrapper.config.log_dir = self.test_dir

        # 4. Run the training process
        model_wrapper.forecast_fit(train_data, train_ratio_in_tv=1.0, trial=None)

        # 5. Assertions
        self.assertTrue(mock_crps_loss.called, "crps_loss was not called during training.")
        # We only care about the calls where the target is normalized for the *training* loss.
        # These calls pass the `base_distr` (a PerChannelDistribution) as the first argument.
        # The logging call passes the `denorm_distr` (a DenormalizingDistribution) and an
        # unclamped, unnormalized target, which we must ignore.
        training_calls_targets = [
            call.args[1] for call in mock_crps_loss.call_args_list
            if isinstance(call.args[0], PerChannelDistribution)
        ]
        self.assertGreater(len(training_calls_targets), 0, "No training calls to crps_loss were captured.")
        full_norm_target_tensor = torch.cat(training_calls_targets, dim=0)

        max_val, min_val = torch.max(full_norm_target_tensor), torch.min(full_norm_target_tensor)
        self.assertTrue(torch.all(max_val <= clip_value),
                        f"Some values in norm_target are > clip value {clip_value}. Max value found: {max_val}")
        self.assertTrue(torch.all(min_val >= -clip_value),
                        f"Some values in norm_target are < clip value {-clip_value}. Min value found: {min_val}")
        
        print(f"\n[Test OK] loss_target_clip correctly clamped values between [{-clip_value}, {clip_value}]. Max seen: {max_val:.4f}, Min seen: {min_val:.4f}")

    @patch('ts_benchmark.baselines.duet.duet_prob.crps_loss')
    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    @patch.object(DUETProb, 'load')
    @patch('torch.save')
    @patch.object(DUETProb, 'validate')
    def test_09_loss_target_clip_is_disabled(self, mock_validate, mock_torch_save, mock_load, mock_SummaryWriter, mock_crps_loss):
        """
        Tests that if `loss_target_clip` is None (disabled), the normalized
        target values are NOT clamped and can exceed typical ranges.
        """
        # 1. Setup mocks
        mock_validate.return_value = (0.5, 0.5) # Match new signature

        # 2. Create data with a massive outlier
        seq_len, horizon, batch_size, n_vars = 4, 8, 4, 2
        # FIX: The mock must return a tensor with the correct shape
        mock_crps_loss.return_value = torch.randn(batch_size, n_vars, horizon, requires_grad=True)

        train_len = seq_len + horizon + batch_size
        dates = pd.date_range(start="2025-01-01", periods=train_len, freq="h")
        data = np.random.randn(train_len, n_vars)
        spike_value, spike_index = 1000.0, seq_len + 2
        data[spike_index, 0] = spike_value
        value_before_spike = data[spike_index - 1, 0]
        train_data = pd.DataFrame(data, index=dates, columns=[f'var_{i}' for i in range(n_vars)])

        # 3. Configure the model with clipping DISABLED
        test_params = self.params.copy()
        test_params.update({
            'loss_target_clip': None, # Explicitly disable
            'norm_mode': 'subtract_last',
            'num_epochs': 1,
            'seq_len': seq_len, 'horizon': horizon, 'batch_size': batch_size
        })
        model_wrapper = DUETProb(**test_params)
        model_wrapper.config.log_dir = self.test_dir

        # 4. Run training
        model_wrapper.forecast_fit(train_data, train_ratio_in_tv=1.0, trial=None)

        # 5. Assertions
        self.assertTrue(mock_crps_loss.called)
        # Filter for training-related calls only, ignoring logging calls with raw targets.
        training_calls_targets = [
            call.args[1] for call in mock_crps_loss.call_args_list
            if isinstance(call.args[0], PerChannelDistribution)
        ]
        self.assertGreater(len(training_calls_targets), 0, "No training calls to crps_loss were captured.")
        full_norm_target_tensor = torch.cat(training_calls_targets, dim=0)

        # We test that the value is large (i.e., normalization of the spike happened)
        # and that this large value wasn't clipped.
        max_val = torch.max(full_norm_target_tensor).item()
        self.assertGreater(max_val, 100.0, "Max value should be large and unclamped after RevIN.")
        print(f"\n[Test OK] loss_target_clip=None correctly passed through a large, unclamped value. Max seen: {max_val:.4f}")
        print(f"\n[Test OK] loss_target_clip=None correctly passed through unclamped value. Max seen: {torch.max(full_norm_target_tensor).item():.4f}")

if __name__ == '__main__':
    # Add a try-except block for the sys.path modification to make the script more robust
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception as e:
        print(f"Could not modify sys.path: {e}")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)