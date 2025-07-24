import unittest
import torch
import sys
import os

# Add project root to the Python path to resolve imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.spliced_binned_pareto_standalone import MLPProjectionHead

class dotdict(dict):
    """A helper class to access dictionary keys via dot notation."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestDUETProbModelProjectionHead(unittest.TestCase):

    def _create_base_config(self) -> dotdict:
        """Creates a minimal, shared configuration for all tests."""
        return dotdict({
            "horizon": 16, "d_model": 32, "d_ff": 64, "n_heads": 4, "e_layers": 1,
            "enc_in": 3, "moving_avg": 25, "dropout": 0.1, "fc_dropout": 0.1,
            "factor": 3, "activation": "gelu", "output_attention": False, "CI": False,
            "num_linear_experts": 1, "num_esn_experts": 1, "hidden_size": 128,
            "k": 1, "noisy_gating": True, "reservoir_size": 64, "spectral_radius": 0.99,
            "sparsity": 0.1, "input_scaling": 1.0, "num_bins": 50, "tail_percentile": 0.05,
            "channel_bounds": {'v1': {}, 'v2': {}, 'v3': {}}, "seq_len": 64,
            # --- HINZUGEFÜGT: Standardwerte für den Projection Head, um den Test zu stabilisieren ---
            "projection_head_layers": 0,
            "projection_head_dim_factor": 2,
            "projection_head_dropout": 0.1,
        })

    def setUp(self):
        """Prepares device and common parameters for tests."""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.B = 4  # Batch size
        self.L = 64 # Sequence length
        self.N_VARS = 3

    def test_01_initialization_with_linear_fallback(self):
        """
        Tests if the model correctly uses the simple linear layer fallback
        when `projection_head_layers` is 0.
        """
        print("\nRunning test: Projection Head Fallback to Linear...")
        config = self._create_base_config()
        config.projection_head_layers = 0 # Explicitly set to 0

        model = DUETProbModel(config).to(self.device)

        # The projection head should be an MLPProjectionHead instance...
        self.assertIsInstance(model.args_proj, MLPProjectionHead)
        # ...but configured to act as a single linear layer.
        self.assertEqual(model.args_proj.num_layers, 0)
        self.assertTrue(hasattr(model.args_proj, 'projection'))
        self.assertFalse(hasattr(model.args_proj, 'residual_blocks'))
        print("OK")

    def test_02_initialization_with_full_mlp_head(self):
        """
        Tests if the model correctly constructs the full residual MLP head
        when `projection_head_layers` is greater than 0.
        """
        print("Running test: Projection Head as full MLP...")
        config = self._create_base_config()
        config.projection_head_layers = 2
        config.projection_head_dim_factor = 2
        config.projection_head_dropout = 0.1

        model = DUETProbModel(config).to(self.device)

        self.assertIsInstance(model.args_proj, MLPProjectionHead)
        self.assertEqual(model.args_proj.num_layers, 2)
        self.assertTrue(hasattr(model.args_proj, 'residual_blocks'))
        self.assertEqual(len(model.args_proj.residual_blocks), 2)
        print("OK")

    def test_03_forward_and_backward_pass_with_mlp_head(self):
        """
        Verifies that the forward and backward passes work correctly with the
        new MLP head, ensuring gradients flow properly.
        """
        print("Running test: Forward/Backward Pass with MLP Head...")
        config = self._create_base_config()
        config.projection_head_layers = 2
        model = DUETProbModel(config).to(self.device)
        model.train()

        input_tensor = torch.randn(self.B, self.L, self.N_VARS).to(self.device)

        # Forward pass
        denorm_distr, _, loss_importance, _, _, _ = model(input_tensor)

        # Create a dummy loss for backpropagation
        # We use the median prediction as a proxy for the distribution's output
        dummy_target = torch.randn_like(denorm_distr.icdf(0.5))
        reconstruction_loss = ((denorm_distr.icdf(0.5) - dummy_target) ** 2).mean()
        total_loss = reconstruction_loss + loss_importance

        # Backward pass
        total_loss.backward()

        # Check for gradients in the last layer of the projection head
        grad = model.args_proj.final_layer.weight.grad
        self.assertIsNotNone(grad, "Gradients should not be None in the projection head.")
        self.assertGreater(torch.abs(grad).sum(), 0, "Sum of gradients should be non-zero, indicating gradient flow.")
        print("OK")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)