import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

# --- Setup project path to allow direct import of the repository's mask ---
try:
    # This assumes the script is in the project's root directory (e.g., 'DUET').
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Use YOUR implementation of the mask from YOUR repository.
    from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask
    print("Successfully imported Mahalanobis_mask from repository.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"FATAL: Could not import Mahalanobis_mask. Ensure this script is in the project root ('DUET'). Error: {e}")
    sys.exit(1)


def calculate_similarity(signal_a, signal_b, mask_instance):
    """
    Helper function to calculate a symmetric similarity score between two signals.
    """
    # Prepare data for the mask: [B, C, L] -> [1, 2, seq_len]
    input_data = torch.stack([signal_a, signal_b], dim=0).unsqueeze(0)
    with torch.no_grad():
        p_learned, _ = mask_instance.calculate_prob_distance(input_data, channel_adjacency_prior=None)
        # The mask calculates p(A->B) and p(B->A). We average them for a symmetric score.
        score_ab = p_learned[0, 0, 1].item()
        score_ba = p_learned[0, 1, 0].item()
        similarity_score = (score_ab + score_ba) / 2
    return similarity_score


def plot_scenario(ax, title, signal_a, signal_b, similarity_score):
    """Helper function to plot a single scenario with its score."""
    ax.plot(signal_a.numpy(), label='Signal A', color='C0', linewidth=2)
    ax.plot(signal_b.numpy(), label='Signal B', color='C1', linestyle='--', linewidth=2)
    
    # Use a color-coded background for the title based on the score
    score_color = plt.get_cmap('viridis')(similarity_score)
    ax.set_title(
        f"{title}\nSimilarity Score: {similarity_score:.4f}",
        fontsize=12, weight='bold',
        bbox=dict(facecolor=score_color, alpha=0.4, edgecolor='none', boxstyle='round,pad=0.4')
    )
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)


def create_frequency_invariance_plot():
    """Generates the plot showing similarity with additive frequencies."""
    # --- Configuration ---
    print("\n--- Generating Frequency Invariance Plot ---") # Fixed: Unterminated string literal
    # A longer sequence length helps to better represent multiple distinct frequencies
    seq_len = 256
    n_vars = 2
    time = torch.linspace(0, 20 * np.pi, seq_len, dtype=torch.float32)

    # --- Instantiate YOUR Mask ---
    mask_instance = Mahalanobis_mask(input_size=seq_len, n_vars=n_vars)

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharex=True)
    fig.suptitle("Mahalanobis_mask Similarity with Additive Frequencies", fontsize=18, weight='bold')
    axes = axes.flatten()

    # --- Define Base Frequencies to build our test signals ---
    # A strong, low-frequency base signal (our "dominant seasonality")
    dominant_freq = torch.sin(time) * 20
    # A medium-frequency signal
    medium_freq = torch.cos(time * 4) * 10
    # Two different high-frequency "noise" signals (non-random)
    additive_freq_a = torch.sin(time * 8 + np.pi / 4) * 5
    additive_freq_b = torch.cos(time * 11) * 7
    # Another completely different low-frequency signal for the control case
    unrelated_freq = torch.sin(time / 2) * 15

    # --- Scenario 1: Single Shared Frequency (Baseline) ---
    print("Running Scenario 1: Single Shared Frequency...")
    s1_a = dominant_freq
    s1_b = dominant_freq.roll(shifts=15, dims=0) + 50 # Shifted in time and value
    score1 = calculate_similarity(s1_a, s1_b, mask_instance)
    plot_scenario(axes[0], "1. Baseline: Single Shared Frequency", s1_a, s1_b, score1)

    # --- Scenario 2: Shared Dominant Frequency + Different Additive Signals ---
    print("Running Scenario 2: Shared Dominant Freq + Different Additive Signals...")
    s2_a = dominant_freq + additive_freq_a
    s2_b = dominant_freq.roll(shifts=15, dims=0) + 50 + additive_freq_b # Base is shared, additive is not
    score2 = calculate_similarity(s2_a, s2_b, mask_instance)
    plot_scenario(axes[1], "2. Shared Dominant Freq + Different Additive Signals", s2_a, s2_b, score2)

    # --- Scenario 3: Multiple Shared Frequencies ---
    print("Running Scenario 3: Multiple Shared Frequencies...")
    s3_a = dominant_freq + medium_freq
    s3_b = (dominant_freq + medium_freq).roll(shifts=15, dims=0) + 50 # Both freqs are shared
    score3 = calculate_similarity(s3_a, s3_b, mask_instance)
    plot_scenario(axes[2], "3. Multiple Shared Frequencies", s3_a, s3_b, score3)

    # --- Scenario 4: Completely Different Frequencies (Control) ---
    print("Running Scenario 4: Different Frequencies (Control)...")
    s4_a = dominant_freq + additive_freq_a
    s4_b = medium_freq + unrelated_freq + 50 # No shared components
    score4 = calculate_similarity(s4_a, s4_b, mask_instance)
    plot_scenario(axes[3], "4. Different Frequencies (Control)", s4_a, s4_b, score4)

    # --- Finalize and Save ---
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_filename = "mask_frequency_invariance.png"
    plt.savefig(output_filename)
    print(f"\nVisualization complete. Plot saved to: {output_filename}")


def run_and_plot_scenario(ax, title, noise_level_base, noise_level_variant):
    """
    Helper function to run and plot a single scenario for similarity with noise,
    y-offset, and time shift.
    """
    # --- Configuration ---
    seq_len = 128
    n_vars = 2

    # --- Create Base Signal (Channel A) ---
    time = torch.linspace(0, 8 * np.pi, seq_len, dtype=torch.float32)
    base_signal_clean = 30 * torch.sin(time) + 20 * torch.cos(time / 2)
    # Add noise to the clean base signal if specified
    base_signal = base_signal_clean + torch.randn(seq_len) * noise_level_base

    # --- Define Ranges for Variants ---
    y_offsets = np.linspace(0, 2000, 30)
    time_shifts = np.linspace(-13, 11, 20, dtype=int)

    # --- Instantiate YOUR Mask ---
    mask_instance = Mahalanobis_mask(input_size=seq_len, n_vars=n_vars)

    # --- Setup colormap ---
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Plot the base signal for reference
    ax.plot(np.arange(seq_len), base_signal.numpy(), color='black', linestyle='--', linewidth=2.5, label='Base Signal (A)', zorder=100)

    # Generate and plot variants
    for y_offset in y_offsets:
        for time_shift in time_shifts:
            # Create the variant signal (Channel B)
            variant_signal_clean = base_signal_clean.roll(shifts=time_shift, dims=0) + y_offset
            # Add noise to the variant signal if specified
            variant_signal = variant_signal_clean + torch.randn(seq_len) * noise_level_variant

            # Prepare data for the mask: [B, C, L] -> [1, 2, seq_len]
            input_data = torch.stack([base_signal, variant_signal], dim=0).unsqueeze(0)

            # Calculate similarity score using your mask
            with torch.no_grad():
                p_learned, _ = mask_instance.calculate_prob_distance(input_data, channel_adjacency_prior=None)
                similarity_score = p_learned[0, 0, 1].item()

            # Plot the variant with color based on its score
            ax.plot(np.arange(seq_len), variant_signal.numpy(), color=cmap(norm(similarity_score)), alpha=0.8)

    # --- Finalize Subplot ---
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # Return cmap and norm for the shared colorbar
    return cmap, norm


if __name__ == "__main__":
    # Call the frequency invariance plot function first to generate its plot
    create_frequency_invariance_plot()

    # --- Create a multi-panel plot to compare scenarios (Time Shift, Y-Offset, Noise) ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
    fig.suptitle("Mahalanobis_mask Similarity Score Under Different Conditions", fontsize=18, weight='bold')

    print("Running Scenario 1: Original (No Noise)...")
    cmap, norm = run_and_plot_scenario(
        axes[0, 0],
        title="Scenario 1: Y-Offset & Time Shift (Original)",
        noise_level_base=0.0,
        noise_level_variant=0.0
    )

    print("Running Scenario 2: Noise on Variant Signal...")
    run_and_plot_scenario(
        axes[0, 1],
        title="Scenario 2: Noise Added to Variant (B)",
        noise_level_base=0.0,
        noise_level_variant=15.0  # A moderate amount of noise
    )

    print("Running Scenario 3: Noise on Both Signals...")
    run_and_plot_scenario(
        axes[1, 0],
        title="Scenario 3: Noise Added to Both (A & B)",
        noise_level_base=15.0,
        noise_level_variant=15.0
    )

    print("Running Scenario 4: High Noise on Variant Signal...")
    run_and_plot_scenario(
        axes[1, 1],
        title="Scenario 4: High Noise on Variant (B)",
        noise_level_base=0.0,
        noise_level_variant=50.0  # A high amount of noise
    )

    # Add a shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Calculated Similarity Score', fontsize=12, weight='bold')

    fig.tight_layout(rect=[0, 0, 0.9, 0.96])

    output_filename = "mask_similarity_scenarios.png"
    plt.savefig(output_filename)
    print(f"\nVisualization complete. Plot saved to: {output_filename}")