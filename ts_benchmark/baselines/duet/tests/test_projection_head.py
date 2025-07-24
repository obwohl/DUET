import unittest
import torch
import torch.nn as nn
import os
import sys

# --- Setup project path to allow direct imports ---
# This ensures the script can find the modules it needs to test.
try:
    # Assumes the test is run from the DUET project root or a similar context.
    # Navigates up from 'tests' -> 'duet' -> 'baselines' -> 'ts_benchmark' -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ts_benchmark.baselines.duet.spliced_binned_pareto_standalone import MLPProjectionHead, SplicedBinnedParetoOutput
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error setting up imports: {e}")
    print("Please ensure you are running tests from the project's root directory.")
    # As a fallback for isolated execution, try a more direct relative import
    from ..spliced_binned_pareto_standalone import MLPProjectionHead, SplicedBinnedParetoOutput


class TestProjectionHead(unittest.TestCase):
    """
    Unit tests for the new MLPProjectionHead and its integration.
    """

    def setUp(self):
        """Set up common parameters for tests."""
        self.batch_size = 4
        self.d_model = 128
        self.args_dim = 104
        self.dummy_input = torch.randn(self.batch_size, self.d_model)

    def test_01_fallback_to_linear_layer(self):
        """
        Tests if the head correctly falls back to a single nn.Linear
        when num_layers is 0. This is the baseline behavior.
        """
        print("\nRunning test: Fallback to Linear Layer...")
        head = MLPProjectionHead(
            in_features=self.d_model, out_features=self.args_dim,
            hidden_dim=64, num_layers=0, dropout=0.1
        )
        # Check that it created a simple linear layer
        self.assertIsInstance(head.projection, nn.Linear)
        self.assertFalse(hasattr(head, 'residual_blocks'))

        output = head(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.args_dim))
        print("OK")

    def test_02_mlp_with_residual_layers(self):
        """
        Tests if the head correctly creates a multi-layer residual MLP
        and that the forward pass works without errors.
        """
        print("Running test: MLP with Residual Layers...")
        head = MLPProjectionHead(
            in_features=self.d_model, out_features=self.args_dim,
            hidden_dim=64, num_layers=2, dropout=0.1
        )
        # Check the internal structure
        self.assertIsInstance(head.residual_blocks, nn.ModuleList)
        self.assertEqual(len(head.residual_blocks), 2)
        self.assertIsInstance(head.final_layer, nn.Linear)

        output = head(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.args_dim))

        # Test if it can be trained (i.e., backward pass works)
        loss = output.sum()
        loss.backward()
        print("OK")

    def test_03_integration_with_splicedbinnedparetooutput(self):
        """
        Tests that the SplicedBinnedParetoOutput class correctly instantiates
        our new MLPProjectionHead via its get_args_proj method.
        """
        print("Running test: Integration with SplicedBinnedParetoOutput...")
        distr_output = SplicedBinnedParetoOutput(num_bins=100, bins_lower_bound=0, bins_upper_bound=1, tail_percentile=0.05, projection_head_layers=1)
        
        projection_module = distr_output.get_args_proj(in_features=self.d_model)
        self.assertIsInstance(projection_module, MLPProjectionHead)
        print("OK")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)