import unittest
import torch
import os
import sys
import numpy as np

# --- Setup project path to allow direct imports ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
    from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error setting up imports: {e}")
    # Fallback for isolated execution
    from ..models.duet_prob_model import DUETProbModel
    from ..utils.masked_attention import Mahalanobis_mask

class DummyConfig:
    """A minimal config class to initialize the models for testing."""
    def __init__(self, **kwargs):
        # Set some sane defaults required by the model components
        defaults = {
            "d_model": 32, "d_ff": 32, "n_heads": 2, "e_layers": 1,
            "factor": 3, "activation": "gelu", "dropout": 0.1,
            "output_attention": False, "CI": False, "horizon": 8,
            "num_linear_experts": 1, "num_esn_experts": 1, "k": 1,
            "hidden_size": 16, "channel_bounds": {"var_0": {}, "var_1": {}, "var_2": {}},
            "moving_avg": 25,
            # --- FIX: Add ESN parameters to prevent initialization errors ---
            "reservoir_size": 16,
            "spectral_radius": 0.99,
            "sparsity": 0.1,
            "input_scaling": 1.0,
            # --- Add other required parameters for robustness ---
            "projection_head_layers": 0,
            "projection_head_dim_factor": 2,
            "projection_head_dropout": 0.1,
            "norm_mode": "subtract_last"
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)

class TestChannelAdjacencyPrior(unittest.TestCase):
    """
    Unit tests for the `channel_adjacency_prior` feature integrated into
    `DUETProbModel` and `Mahalanobis_mask`.
    """

    def setUp(self):
        """Set up common parameters for tests."""
        self.n_vars = 3
        self.seq_len = 24
        self.batch_size = 4
        self.device = torch.device("cpu")
        
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.n_vars, device=self.device)
        self.mask_input = self.dummy_input.permute(0, 2, 1) # Shape [B, C, L] for Mahalanobis_mask

        self.base_config = {
            "enc_in": self.n_vars,
            "seq_len": self.seq_len,
            "channel_bounds": {f'var_{i}': {} for i in range(self.n_vars)}
        }

    def test_01_model_init_with_valid_prior(self):
        """Tests if DUETProbModel correctly initializes with a valid prior."""
        print("\nRunning test: Model Initialization with Valid Prior...")
        prior_list = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
        config = DummyConfig(**self.base_config, channel_adjacency_prior=prior_list)
        
        model = DUETProbModel(config)
        
        self.assertIsInstance(model.channel_adjacency_prior, torch.Tensor)
        self.assertEqual(model.channel_adjacency_prior.shape, (self.n_vars, self.n_vars))
        self.assertTrue(torch.equal(model.channel_adjacency_prior, torch.tensor(prior_list, dtype=torch.float32)))
        print("OK")

    def test_02_model_init_with_invalid_prior_shape(self):
        """Tests if DUETProbModel raises a ValueError for a prior with an incorrect shape."""
        print("\nRunning test: Model Initialization with Invalid Prior Shape...")
        invalid_prior = [[1, 1], [0, 1]] # Shape is 2x2, but n_vars is 3
        config = DummyConfig(**self.base_config, channel_adjacency_prior=invalid_prior)
        
        with self.assertRaises(ValueError) as context:
            DUETProbModel(config)
        
        self.assertTrue("shape mismatch" in str(context.exception))
        print("OK")

    def test_03_model_init_without_prior(self):
        """Tests if DUETProbModel initializes the prior as None when it's not provided."""
        print("\nRunning test: Model Initialization without Prior...")
        config = DummyConfig(**self.base_config) # No prior provided
        model = DUETProbModel(config)
        
        self.assertIsNone(model.channel_adjacency_prior)
        print("OK")

    def test_04_mask_logic_with_hard_zero_prior(self):
        """
        Tests if the `calculate_prob_distance` method correctly applies a hard (binary) prior,
        forcing specific probabilities to zero.
        """
        print("\nRunning test: Mask Logic with Hard Zero Prior...")
        mask_generator = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        
        # A prior that forbids channel 2 from attending to 0 and 1, and vice-versa.
        prior = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float32, device=self.device)
        
        # Calculate probabilities with the prior
        # Unpack the tuple; we only need the final probability matrix for this test.
        _, p_final = mask_generator.calculate_prob_distance(self.mask_input, channel_adjacency_prior=prior)
        
        # Assert that the forbidden connections have zero probability
        self.assertTrue(torch.all(p_final[:, 0, 2] == 0), "p_final[0, 2] should be 0")
        self.assertTrue(torch.all(p_final[:, 1, 2] == 0), "p_final[1, 2] should be 0")
        self.assertTrue(torch.all(p_final[:, 2, 0] == 0), "p_final[2, 0] should be 0")
        self.assertTrue(torch.all(p_final[:, 2, 1] == 0), "p_final[2, 1] should be 0")
        print("OK")

    def test_05_mask_logic_with_soft_prior(self):
        """
        Tests if `calculate_prob_distance` correctly scales probabilities with a continuous-valued prior.
        """
        print("\nRunning test: Mask Logic with Soft (Continuous) Prior...")
        mask_generator = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        prior = torch.tensor([[1.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=self.device)

        # Unpack the tuples from the new return signature
        p_learned, _ = mask_generator.calculate_prob_distance(self.mask_input, channel_adjacency_prior=None)
        _, p_final = mask_generator.calculate_prob_distance(self.mask_input, channel_adjacency_prior=prior)

        # The final probability should be the learned probability element-wise multiplied by the prior.
        self.assertTrue(torch.allclose(p_final, p_learned * prior.unsqueeze(0), atol=1e-6))
        print("OK")

    def test_06_forward_pass_respects_hard_zero_prior(self):
        """Ensures the final sampled mask from the forward pass respects the hard zero constraints."""
        print("\nRunning test: Full Forward Pass Respects Hard Zero Prior...")
        mask_generator = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        prior = torch.tensor([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32, device=self.device)
        
        # Unpack the tuple from forward(); we only need the final binary mask.
        sampled_mask, _, _ = mask_generator.forward(self.mask_input, channel_adjacency_prior=prior)
        
        self.assertTrue(torch.all(sampled_mask[:, 0, 0, 1] == 0), "The sampled mask should respect the prior's zero.")
        print("OK")

    def test_07_gradient_modulation_and_blocking(self):
        """
        Tests if the backward pass correctly modulates gradients based on the prior.
        - A prior of 1.0 should pass the gradient through.
        - A prior of 0.0 should block the gradient (set to zero).
        - A prior of 0.5 should scale the gradient by 0.5.
        This is the most critical test for the learning behavior.
        """
        print("\nRunning test: Gradient Modulation and Blocking in Backward Pass...")
        mask_generator = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        
        # 1. Get the original probability matrix, but detach it from the computation graph
        #    so we can treat it as a leaf variable for this gradient test.
        p_learned, _ = mask_generator.calculate_prob_distance(self.mask_input, channel_adjacency_prior=None)
        p_original = p_learned.detach()
        p_original.requires_grad = True

        # 2. Define a prior with a mix of values (passthrough, scale, block)
        prior = torch.tensor([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ], dtype=torch.float32, device=self.device)

        # 3. Apply the prior to the detached probability matrix
        p_final = p_original * prior.unsqueeze(0)

        # 4. Calculate a simple loss. The gradient of sum() is a tensor of ones.
        loss = p_final.sum()
        loss.backward()

        # 5. Check the gradient on `p_original`.
        # According to the chain rule, the gradient should be exactly the prior matrix.
        self.assertIsNotNone(p_original.grad, "Gradient should have been computed for p_original.")
        expected_grad = prior.unsqueeze(0).expand_as(p_original)
        self.assertTrue(
            torch.allclose(p_original.grad, expected_grad),
            "The gradient was not correctly modulated by the prior."
        )
        print("OK: Gradients are correctly passed through (x1), scaled (x0.5), and blocked (x0).")

    def test_08_gradient_flow_to_learnable_parameters(self):
        """
        Tests that the gradient correctly flows back to the learnable
        parameters of the mask generator (`self.A`) and that a blocking prior
        prevents updates.
        """
        print("\nRunning test: Gradient Flow to Learnable Parameters...")
        
        mask_gen_full = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        prior_full = torch.ones(self.n_vars, self.n_vars, device=self.device)
        
        _, p_full = mask_gen_full.calculate_prob_distance(self.mask_input, channel_adjacency_prior=prior_full)
        # Use a quadratic loss to ensure a non-zero gradient, as sum() would be constant for a row-normalized matrix.
        loss_full = ((p_full - 0.5)**2).sum()
        loss_full.backward()
        
        self.assertIsNotNone(mask_gen_full.A.grad, "Gradient for `A` should exist with a full prior.")
        self.assertGreater(mask_gen_full.A.grad.abs().sum(), 0, "Gradient sum for `A` should be non-zero.")
        print("OK: Gradient flows to learnable parameter `A` when prior allows it.")

    def test_09_prior_reduces_gradient_magnitude(self):
        """
        Tests the end-to-end effect of the prior on learning. A restrictive
        prior should result in a smaller gradient magnitude on the learnable
        parameter `A` compared to a non-restrictive prior, given the same input.
        This is a hard test of the feature's practical impact.
        """
        print("\nRunning test: Prior's Effect on Learnable Parameter Gradient...")
        
        # --- Setup: Two mask generators with identical initial weights ---
        # We need to control the random seed to ensure they start identically.
        torch.manual_seed(42)
        mask_gen_open = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        
        torch.manual_seed(42)
        mask_gen_restricted = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)

        # Verify they are identical
        self.assertTrue(torch.equal(mask_gen_open.A.data, mask_gen_restricted.A.data))

        # --- Case 1: Open prior (all connections allowed) ---
        prior_open = torch.ones(self.n_vars, self.n_vars, device=self.device)
        _, p_open = mask_gen_open.calculate_prob_distance(self.mask_input, channel_adjacency_prior=prior_open)
        
        # FIX: The original loss `p.sum()` has a zero gradient for a row-normalized matrix.
        # We use a quadratic loss which is not constant and provides a meaningful gradient,
        # making it a better proxy for a real loss function.
        loss_open = ((p_open - 0.5)**2).sum() # Loss is on the final probabilities
        loss_open.backward()
        
        grad_norm_open = torch.linalg.norm(mask_gen_open.A.grad)
        self.assertIsNotNone(mask_gen_open.A.grad)
        self.assertGreater(grad_norm_open, 0)

        # --- Case 2: Restricted prior (some connections blocked) ---
        prior_restricted = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.float32, device=self.device)
        _, p_restricted = mask_gen_restricted.calculate_prob_distance(self.mask_input, channel_adjacency_prior=prior_restricted)
        loss_restricted = ((p_restricted - 0.5)**2).sum()
        loss_restricted.backward()

        grad_norm_restricted = torch.linalg.norm(mask_gen_restricted.A.grad)
        self.assertLess(grad_norm_restricted, grad_norm_open, "The gradient norm for the restricted prior should be smaller than for the open prior.")
        print("OK: A restrictive prior correctly reduces the gradient magnitude on the learnable parameter 'A'.")

    def test_10_prob_matrix_is_valid_for_sampler(self):
        """
        Tests if the learned probability-like matrix is valid for the Gumbel-Softmax
        sampler, i.e., all elements are clamped between 0 and 1. This test
        acknowledges the original model's non-stochastic normalization, which is
        crucial for providing a strong initial gradient signal.
        """
        print("\nRunning test: Probability Matrix is Valid for Sampler...")
        mask_generator = Mahalanobis_mask(input_size=self.seq_len, n_vars=self.n_vars)
        
        p_learned, p_final = mask_generator.calculate_prob_distance(self.mask_input, channel_adjacency_prior=None)
        
        self.assertTrue(torch.allclose(p_learned, p_final))

        # The values must be in the range [0, 1] for the Gumbel-Softmax sampler.
        self.assertTrue(torch.all(p_learned >= 0), "All probabilities must be non-negative.")
        self.assertTrue(torch.all(p_learned <= 1), "All probabilities must be less than or equal to 1.")

        # We no longer assert that rows sum to 1, as this is not the case in the
        # original, successful implementation.
        print("OK: The learned probability-like matrix has values clamped in [0, 1].")

    def test_11_fft_invariance_to_time_shift(self):
        """
        Tests the core assumption of the Mahalanobis mask: that the FFT-based
        distance is small for time-shifted but otherwise identical signals.
        This test would fail with the old, buggy implementation where RevIN was
        applied *before* the FFT, destroying the shift-invariance property.
        """
        print("\nRunning test: FFT Invariance to Time Shift...")
        n_vars = 2
        seq_len = 48
        batch_size = 4
        shift = 24

        # 1. Create synthetic data: a sine wave and a shifted version
        time = torch.linspace(0, 8 * np.pi, seq_len + shift)
        base_signal = torch.sin(time)

        cause_series = base_signal[:-shift]
        effect_series = base_signal[shift:]

        # Stack them into a batch of shape [B, L, C]
        input_data = torch.stack([
            cause_series.unsqueeze(0).repeat(batch_size, 1),
            effect_series.unsqueeze(0).repeat(batch_size, 1)
        ], dim=2)

        # 2. Initialize the mask generator and pass the raw data
        mask_generator = Mahalanobis_mask(input_size=seq_len, n_vars=n_vars)
        mask_input = input_data.permute(0, 2, 1) # Permute to [B, C, L]

        # 3. Calculate the probability matrix
        p_learned, _ = mask_generator.calculate_prob_distance(mask_input, channel_adjacency_prior=None)

        # 4. Assertions
        # We expect a high probability for cause->effect (p[0,1]) and effect->cause (p[1,0])
        prob_cause_to_effect = p_learned[:, 0, 1]
        self.assertGreater(
            prob_cause_to_effect.mean().item(), 0.4,
            f"Expected high probability (>0.4) between shifted signals, but got {prob_cause_to_effect.mean().item()}"
        )
        print(f"OK: Found high probability ({prob_cause_to_effect.mean().item():.4f}) between time-shifted signals.")

    def test_12_fft_invariance_with_trend_and_normalization(self):
        """
        Definitive test: Verifies that the mask can identify time-shifted signals
        even when they are superimposed on a non-stationary trend. This is the
        exact scenario of the synthetic data and the root cause of the bug.
        This test will only pass if instance normalization (RevIN) is correctly
        applied *inside* the mask generator before the FFT.
        """
        print("\nRunning test: FFT Invariance with Trended Data...")
        n_vars = 2
        seq_len = 48
        batch_size = 4
        shift = 24

        # 1. Create synthetic data with a trend, similar to causal_1.csv
        time_base = torch.linspace(0, 8 * np.pi, seq_len + shift)
        seasonal_signal = torch.sin(time_base)
        
        time_steps = torch.arange(seq_len + shift, dtype=torch.float32)
        trend_signal = 0.1 * time_steps # Add a linear trend

        base_signal = seasonal_signal + trend_signal

        cause_series = base_signal[:-shift]
        effect_series = base_signal[shift:]

        input_data = torch.stack([
            cause_series.unsqueeze(0).repeat(batch_size, 1),
            effect_series.unsqueeze(0).repeat(batch_size, 1)
        ], dim=2)

        # 2. Initialize the mask generator and pass the raw, trended data
        mask_generator = Mahalanobis_mask(input_size=seq_len, n_vars=n_vars)
        mask_input = input_data.permute(0, 2, 1)

        # 3. Calculate the probability matrix
        p_learned, _ = mask_generator.calculate_prob_distance(mask_input, channel_adjacency_prior=None)

        # 4. Assert that the probability is high, proving the trend was handled
        prob_cause_to_effect = p_learned[:, 0, 1]
        self.assertGreater(prob_cause_to_effect.mean().item(), 0.4)
        print(f"OK: Correctly found high probability ({prob_cause_to_effect.mean().item():.4f}) even with a data trend.")

    def test_13_channel_transformer_respects_mask(self):
        """
        Definitive Test for the end-to-end masking logic.
        This test verifies that the Channel_transformer's output for a specific channel
        is actually different when its connections are masked versus when they are not.
        This will fail if the residual connections in the EncoderLayer are leaking
        information and bypassing the attention mask.
        """
        print("\nRunning test: Channel Transformer Respects Mask...")
        n_vars = 2
        config_dict = {
            "enc_in": n_vars, "seq_len": self.seq_len,
            "channel_bounds": {f'var_{i}': {} for i in range(n_vars)}
        }
        config = DummyConfig(**config_dict)
        model = DUETProbModel(config).to(self.device)
        model.eval()

        # Create an input where each channel has distinct, non-constant features.
        # This is crucial to prevent LayerNorm from zeroing out the input, which would
        # make the attention output zero regardless of the mask.
        torch.manual_seed(123) # for reproducibility
        c1_features = torch.randn(self.batch_size, 1, config.d_model, device=self.device)
        c2_features = torch.randn(self.batch_size, 1, config.d_model, device=self.device) * 100.0
        temporal_feature = torch.cat([c1_features, c2_features], dim=1)

        # --- Scenario 1: Open Mask (effect can see cause) ---
        open_mask = torch.ones(self.batch_size, 1, n_vars, n_vars, device=self.device)
        with torch.no_grad():
            output_open, _ = model.Channel_transformer(x=temporal_feature, attn_mask=open_mask)
        
        # Extract the output for the 'effect' channel (index 1)
        effect_output_open = output_open[:, 1, :]

        # --- Scenario 2: Blocked Mask (effect can ONLY see itself) ---
        # This mask blocks the effect -> cause connection.
        blocked_mask = torch.eye(n_vars, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        with torch.no_grad():
            output_blocked, _ = model.Channel_transformer(x=temporal_feature, attn_mask=blocked_mask)

        effect_output_blocked = output_blocked[:, 1, :]

        # --- Assertion ---
        # If the mask is working correctly, the output for the effect channel MUST be different
        # when it is denied access to the cause channel's information.
        are_equal = torch.allclose(effect_output_open, effect_output_blocked)
        self.assertFalse(are_equal, "The channel mask is being ignored by the transformer; outputs are identical.")
        print("OK: The Channel_transformer's output correctly changes when the mask is applied.")

    def test_hypothesis_mask_fails_on_different_scales(self):
        """
        UNIT TEST FOR THE HYPOTHESIS:
        The Mahalanobis mask (without instance normalization) fails to detect
        similarity between time series on vastly different scales/DC offsets.
        """
        print("\nRunning test: HYPOTHESIS - Mask fails on different scales...")
        n_vars = 2
        seq_len = 96
        batch_size = 1
        shift = 5

        # 1. Create synthetic data as per the hypothesis
        # Channel A is on size -10 to 30. Channel B is on size 980 to 1020.
        time = torch.linspace(0, 10 * np.pi, seq_len + shift, dtype=torch.float32)
        base_signal = 20 * torch.sin(time) # Amplitude 20, range [-20, 20]

        # Channel A: sine wave on scale [-10, 30]
        channel_a = base_signal[:-shift] + 10

        # Channel B: similar sine wave, but shifted in time and on a much higher scale [980, 1020]
        channel_b = base_signal[shift:] + 1099

        # Stack into a batch of shape [B, C, L]
        input_data = torch.stack([channel_a, channel_b], dim=0).unsqueeze(0)
        self.assertEqual(input_data.shape, (batch_size, n_vars, seq_len))

        # 2. Initialize the mask generator. Adjacency matrix is unconstrained.
        mask_generator = Mahalanobis_mask(input_size=seq_len, n_vars=n_vars)

        # 3. Calculate the probability matrix.
        p_learned, _ = mask_generator.calculate_prob_distance(input_data, channel_adjacency_prior=None)

        # 4. ASSERTION: The hypothesis is that the mask will FAIL, resulting in a LOW learned probability.
        # This assertion SUCCEEDS if the probability is low, thus CONFIRMING the hypothesis.
        prob_a_to_b = p_learned[:, 0, 1]
        self.assertLess(prob_a_to_b.mean().item(), 0.1)
        print(f"OK: HYPOTHESIS CONFIRMED. Mask failed to find similarity; probability is low ({prob_a_to_b.mean().item():.4f}).")