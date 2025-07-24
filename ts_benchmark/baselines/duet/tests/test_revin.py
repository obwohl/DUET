import unittest
import torch
# Make sure your import statement matches your file structure
from ..layers.RevIN import RevIN 

class TestRevIN(unittest.TestCase):

    def setUp(self):
        """Set up common tensors for all tests."""
        self.batch_size = 8
        self.seq_len = 20
        self.num_features = 4
        # Create some non-trivial data with a trend
        self.input_data = torch.linspace(0, 1, self.seq_len).unsqueeze(0).unsqueeze(-1)
        self.input_data = self.input_data.repeat(self.batch_size, 1, self.num_features)
        self.input_data += torch.randn(self.batch_size, self.seq_len, self.num_features) * 0.1

    def _test_reversibility(self, norm_mode, affine):
        """Helper function to test the norm -> denorm cycle."""
        revin_layer = RevIN(self.num_features, norm_mode=norm_mode, affine=affine)
        
        norm_data, stats = revin_layer(self.input_data, mode='norm')
        denorm_data = revin_layer(norm_data, mode='denorm')
        
        self.assertTrue(
            torch.allclose(self.input_data, denorm_data, atol=1e-5),
            f"Reversibility failed for norm_mode='{norm_mode}' with affine={affine}"
        )

    def test_01_reversibility_all_modes_no_affine(self):
        """Tests the norm -> denorm cycle WITHOUT affine transform for all modes."""
        print("\nRunning test: Reversibility (non-affine)...")
        for mode in ['identity', 'subtract_last', 'subtract_median']:
            print(f"  - Testing mode: {mode}")
            self._test_reversibility(norm_mode=mode, affine=False)
            
    def test_02_reversibility_all_modes_affine(self):
        """Tests the norm -> denorm cycle WITH affine transform for all modes."""
        print("\nRunning test: Reversibility (affine)...")
        for mode in ['identity', 'subtract_last', 'subtract_median']:
            print(f"  - Testing mode: {mode}")
            self._test_reversibility(norm_mode=mode, affine=True)

    def test_03_identity_mode(self):
        """Verifies that 'identity' mode without affine does not change the data."""
        print("\nRunning test: Identity Mode Correctness...")
        revin_layer = RevIN(self.num_features, norm_mode='identity', affine=False)
        norm_data, stats = revin_layer(self.input_data, mode='norm')
        self.assertTrue(torch.allclose(stats[..., 0], torch.zeros(1)))
        self.assertTrue(torch.allclose(stats[..., 1], torch.ones(1)))
        self.assertTrue(torch.allclose(self.input_data, norm_data))

    def test_04_subtract_last_correctness(self):
        """Verifies the statistics calculation for 'subtract_last' mode."""
        print("\nRunning test: 'subtract_last' Stats Correctness...")
        revin_layer = RevIN(self.num_features, norm_mode='subtract_last', affine=False)
        _, stats = revin_layer(self.input_data, mode='norm')
        
        expected_location = self.input_data[:, -1, :]
        centered_data = self.input_data - expected_location.unsqueeze(1)
        
        # --- FIX: Added the layer's 'eps' to the calculation for a perfect match ---
        expected_scale = torch.sqrt(torch.mean(centered_data**2, dim=1) + revin_layer.eps)

        self.assertTrue(torch.allclose(stats[..., 0], expected_location, atol=1e-6))
        self.assertTrue(torch.allclose(stats[..., 1], expected_scale, atol=1e-6))

    def test_05_subtract_median_correctness(self):
        """Verifies the statistics calculation for 'subtract_median' mode."""
        print("\nRunning test: 'subtract_median' Stats Correctness...")
        revin_layer = RevIN(self.num_features, norm_mode='subtract_median', affine=False)
        _, stats = revin_layer(self.input_data, mode='norm')
        
        expected_location = torch.median(self.input_data, dim=1)[0]
        centered_data = self.input_data - expected_location.unsqueeze(1)
        
        # --- FIX: Added the layer's 'eps' to the calculation for a perfect match ---
        expected_scale = torch.sqrt(torch.mean(centered_data**2, dim=1) + revin_layer.eps)

        self.assertTrue(torch.allclose(stats[..., 0], expected_location, atol=1e-6))
        self.assertTrue(torch.allclose(stats[..., 1], expected_scale, atol=1e-6))

    def test_06_denorm_with_external_stats(self):
        """Verifies that de-normalization works with externally provided stats."""
        print("\nRunning test: De-normalization with External Stats...")
        
        # --- FIX: This test now uses affine=False to correctly isolate the 'stats_to_use' feature ---
        revin_layer1 = RevIN(self.num_features, norm_mode='subtract_last', affine=False)
        norm_data, stats = revin_layer1(self.input_data, mode='norm')
        
        revin_layer2 = RevIN(self.num_features, norm_mode='subtract_last', affine=False)
            
        # De-normalize using the stats from the first layer
        denorm_data = revin_layer2(norm_data, mode='denorm', stats_to_use=stats)
        
        self.assertTrue(
            torch.allclose(self.input_data, denorm_data, atol=1e-5),
            "De-normalization with external stats failed."
        )

if __name__ == '__main__':
    unittest.main()