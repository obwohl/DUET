import unittest
import torch
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# --- Setup project path to allow direct imports ---
try:
    # Attempt to import from the project structure
    from ts_benchmark.baselines.duet.duet_prob import DUETProb
    from ts_benchmark.baselines.utils import train_val_split, forecasting_data_provider
except (ImportError, ModuleNotFoundError):
    # Fallback for local execution if the above fails
    print("Could not import DUETProb from the expected project path. This test may fail.")
    # This assumes a different relative path, adjust if necessary
    from ..duet_prob import DUETProb
    from ...utils import train_val_split, forecasting_data_provider


class TestDUETProbIntegration(unittest.TestCase):
    """
    A consolidated test suite for the DUETProb wrapper, covering new visualization
    features, training loop logic, and regression testing for signature changes.
    """

    def setUp(self):
        """Set up a minimal environment for testing the DUETProb wrapper."""
        self.test_dir = "temp_integration_test_runs"
        os.makedirs(self.test_dir, exist_ok=True)

        self.params = {
            "seq_len": 16, "horizon": 8, "d_model": 8, "d_ff": 8, "n_heads": 1,
            "e_layers": 1, "num_linear_experts": 1, "num_esn_experts": 1, "k": 1,
            "num_epochs": 1, "batch_size": 4, "patience": 3, "lr": 1e-4,
            "quantiles": [0.1, 0.5, 0.9], "reservoir_size": 8,
            "channel_adjacency_prior": None,  # Explicitly set for tests
            "log_dir": self.test_dir
        }

        self.n_vars = 3
        train_len = 200
        dates = pd.date_range(start="2025-01-01", periods=train_len, freq="h")
        data = np.random.randn(train_len, self.n_vars)
        self.train_valid_data = pd.DataFrame(data, index=dates, columns=[f'var_{i}' for i in range(self.n_vars)])

    def tearDown(self):
        """Clean up temporary directories after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        plt.close('all')  # Close all matplotlib figures

    # --- 1. Regression Tests for Method Signature Changes ---

    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    def test_01_forecast_method_handles_new_signature(self, mock_writer):
        """Tests if the `forecast` method runs without crashing after the model's return signature changed."""
        print("\nRunning test: `forecast` method regression...")
        model_wrapper = DUETProb(**self.params)
        model_wrapper.forecast_fit(self.train_valid_data, train_ratio_in_tv=1.0)

        try:
            forecast_output = model_wrapper.forecast(
                horizon=self.params['horizon'],
                train=self.train_valid_data.iloc[-self.params['seq_len']:]
            )
            self.assertIsInstance(forecast_output, np.ndarray)
            self.assertEqual(forecast_output.shape, (self.n_vars, self.params['horizon'], len(self.params['quantiles'])))
            print("OK: `forecast` method executed successfully.")
        except Exception as e:
            self.fail(f"`forecast` method failed with an unexpected error: {e}")

    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._create_window_plot', return_value=Figure())
    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    def test_02_log_interesting_window_plots_handles_new_signature(self, mock_writer, mock_create_plot):
        """Tests if `_log_interesting_window_plots` runs without crashing after the signature change."""
        print("\nRunning test: `_log_interesting_window_plots` regression...")
        model_wrapper = DUETProb(**self.params)
        model_wrapper.forecast_fit(self.train_valid_data, train_ratio_in_tv=0.8)

        model_wrapper.interesting_window_indices = {'var_0': {'max_change': 10}}
        
        _, valid_data = train_val_split(self.train_valid_data, 0.8, model_wrapper.config.seq_len)
        valid_dataset, _ = forecasting_data_provider(
            valid_data, model_wrapper.config, timeenc=1, batch_size=1,
            shuffle=False, drop_last=False
        )

        try:
            model_wrapper._log_interesting_window_plots(epoch=1, writer=mock_writer.return_value, valid_dataset=valid_dataset)
            print("OK: `_log_interesting_window_plots` executed successfully.")
        except Exception as e:
            self.fail(f"`_log_interesting_window_plots` failed with an unexpected error: {e}")

    # --- 2. Tests for New Visualization Components ---

    def test_03_plot_single_heatmap_helper(self):
        """Tests the `_plot_single_heatmap` helper function's drawing logic."""
        print("\nRunning test: `_plot_single_heatmap` helper...")
        model_wrapper = DUETProb(**self.params)
        fig, ax = plt.subplots()
        
        matrix = np.array([[0.1, 0.9], [0.5, 0.6]])
        channel_names = ['A', 'B']
        title = "Test Heatmap"

        model_wrapper._plot_single_heatmap(ax, matrix, title, channel_names)

        self.assertEqual(ax.get_title(), title)
        self.assertEqual(len(ax.texts), 4)
        
        text_colors = sorted([t.get_c() for t in ax.texts])
        self.assertIn('black', text_colors, "Black text should be used for high-value cells.")
        self.assertIn('w', text_colors, "White text should be used for low-value cells.")
        print("OK: `_plot_single_heatmap` creates correct plot elements.")

    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._plot_single_heatmap')
    def test_04_log_dependency_heatmaps_main_plotter(self, mock_plot_single):
        """Tests the main 3-panel plotting function `_log_dependency_heatmaps`."""
        print("\nRunning test: `_log_dependency_heatmaps` main plotter...")
        model_wrapper = DUETProb(**self.params)
        dummy_matrix = np.random.rand(self.n_vars, self.n_vars)

        fig_dummy, ax_dummy = plt.subplots()
        dummy_image_mappable = ax_dummy.imshow(np.zeros((2, 2)))
        plt.close(fig_dummy) 
        mock_plot_single.return_value = dummy_image_mappable

        prior_tensor = torch.rand(self.n_vars, self.n_vars)
        fig = model_wrapper._log_dependency_heatmaps(prior_tensor, dummy_matrix.copy(), dummy_matrix.copy())

        self.assertIsInstance(fig, Figure)
        self.assertEqual(mock_plot_single.call_count, 3)
        
        expected_titles = ["User Prior", "Learned (Unconstrained)", "Effective (Constrained)"]
        actual_titles = [c.args[2] for c in mock_plot_single.call_args_list]
        self.assertListEqual(actual_titles, expected_titles)

        mock_plot_single.reset_mock()
        mock_plot_single.return_value = dummy_image_mappable
        model_wrapper._log_dependency_heatmaps(None, dummy_matrix.copy(), dummy_matrix.copy())
        
        prior_call_args = mock_plot_single.call_args_list[0]
        received_prior_matrix = prior_call_args.args[1]
        self.assertTrue(np.all(received_prior_matrix == 1.0))
        print("OK: `_log_dependency_heatmaps` calls helpers correctly.")
    
    @patch('ts_benchmark.baselines.duet.duet_prob.crps_loss')
    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._log_dependency_heatmaps')
    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProbModel')
    def test_05_matrix_collection_and_averaging(self, mock_model_class, mock_writer, mock_log_heatmaps, mock_crps_loss):
        """Tests if the probability matrices are correctly collected and averaged over an epoch."""
        print("\nRunning test: Matrix collection and averaging...")

        # --- Systematically Constructed Mocks ---

        # 1. Mock for the DUETProbModel class and its methods
        mock_model_instance = mock_model_class.return_value
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.state_dict.return_value = {"dummy_param": torch.nn.Parameter(torch.rand(1))}
        mock_model_instance.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.rand(1))])
        
        # 2. Mock for the distribution object returned by the model's forward pass
        mock_dist = MagicMock()
        mock_dist.mean.device = torch.device('cpu')
        
        shape_icdf = (1, self.params['horizon'], self.n_vars, len(self.params['quantiles']))
        mock_dist.icdf.return_value = torch.rand(shape_icdf)
        
        shape_norm = (self.params['batch_size'], self.params['horizon'], self.n_vars)
        mock_dist.normalize_value.return_value = torch.rand(shape_norm)

        p_learned = torch.rand(self.params['batch_size'], self.n_vars, self.n_vars)
        p_final = torch.rand(self.params['batch_size'], self.n_vars, self.n_vars)
        dummy_tensor = torch.rand(1)
        
        mock_model_instance.side_effect = [
            (mock_dist, mock_dist, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, p_learned, p_final)
        ] * 20  # Provide more than enough mock returns to be safe

        # 3. Use a robust callable function for the crps_loss side_effect.
        def robust_crps_mock(*args, **kwargs):
            y_true = args[1]
            return torch.full(y_true.shape, 0.5, requires_grad=True, device=y_true.device)

        mock_crps_loss.side_effect = robust_crps_mock

        # --- Run Training ---
        model_wrapper = DUETProb(**self.params)
        
        num_samples = 40
        small_train_data = self.train_valid_data.iloc[:num_samples]
        
        with patch.object(model_wrapper, 'validate', return_value=(0.5, 0.5)):
            model_wrapper.forecast_fit(small_train_data, train_ratio_in_tv=0.8)

        # --- Assertions ---
        self.assertTrue(mock_log_heatmaps.called, "Plotting function for dependency heatmaps was not called.")
        
        call_kwargs = mock_log_heatmaps.call_args.kwargs
        avg_learned_matrix = call_kwargs['learned_matrix']
        self.assertIsNotNone(avg_learned_matrix, "Averaged learned matrix should not be None.")
        print("OK: Test completed and assertions passed.")



    @patch('ts_benchmark.baselines.duet.duet_prob.DUETProb._log_dependency_heatmaps')
    @patch('ts_benchmark.baselines.duet.duet_prob.SummaryWriter')
    def test_06_conditional_logging_trigger(self, mock_writer, mock_log_heatmaps):
        """Tests that the dependency heatmap is only logged when validation loss improves."""
        print("\nRunning test: Conditional logging trigger...")
        model_wrapper = DUETProb(**self.params)
        
        # --- Scenario 1: Validation loss improves ---
        print("  - Scenario 1: Loss improves...")
        with patch.object(model_wrapper, 'validate', return_value=(0.5, 0.5)):
            with patch('ts_benchmark.baselines.duet.duet_prob.EarlyStopping') as mock_es:
                mock_es_instance = mock_es.return_value
                mock_es_instance.val_loss_min = 1.0
                mock_es_instance.early_stop = False
                mock_es_instance.check_point = None

                def early_stopping_side_effect(loss, model_state):
                    if loss < mock_es_instance.val_loss_min:
                        mock_es_instance.val_loss_min = loss
                mock_es_instance.side_effect = early_stopping_side_effect

                model_wrapper.forecast_fit(self.train_valid_data, train_ratio_in_tv=0.8)
                self.assertTrue(mock_log_heatmaps.called, "Plotting should be called on improvement.")
        print("    OK")

        # --- Scenario 2: Validation loss does NOT improve ---
        print("  - Scenario 2: Loss does not improve...")
        mock_log_heatmaps.reset_mock()
        model_wrapper = DUETProb(**self.params)
        with patch.object(model_wrapper, 'validate', return_value=(1.5, 1.5)):
            with patch('ts_benchmark.baselines.duet.duet_prob.EarlyStopping') as mock_es:
                mock_es_instance = mock_es.return_value
                mock_es_instance.val_loss_min = 1.0
                mock_es_instance.early_stop = False
                mock_es_instance.check_point = None
                model_wrapper.forecast_fit(self.train_valid_data, train_ratio_in_tv=0.8)
                self.assertFalse(mock_log_heatmaps.called, "Plotting should NOT be called if loss does not improve.")
        print("    OK")

        # --- Scenario 3: No validation data ---
        print("  - Scenario 3: No validation data...")
        mock_log_heatmaps.reset_mock()
        model_wrapper = DUETProb(**self.params)
        model_wrapper.forecast_fit(self.train_valid_data, train_ratio_in_tv=1.0)
        self.assertFalse(mock_log_heatmaps.called, "Plotting should NOT be called if there is no validation set.")
        print("    OK")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)