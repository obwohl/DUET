# ts_benchmark/baselines/duet/utils/crps.py (Version 2)

import torch
from torch.distributions import Distribution

def crps_loss(
    distr: Distribution, 
    y_true: torch.Tensor, 
    num_quantiles: int = 99
) -> torch.Tensor:
    """
    Berechnet den Continuous Ranked Probability Score (CRPS) durch eine
    Approximation über den Pinball Loss einer großen Anzahl von Quantilen.

    Diese Implementierung ist vollständig vektorisiert und effizient.

    Args:
        distr (Distribution): Eine PyTorch Distribution-Instanz. Es wird erwartet,
                              dass diese eine `.icdf()`-Methode (inverse CDF) hat.
        y_true (torch.Tensor): Die wahren Zielwerte. Erwartet in der Form [B, N_vars, H].
        num_quantiles (int): Die Anzahl der Quantile, die zur Approximation
                             verwendet werden (z.B. 1%, 2%, ..., 99%).

    Returns:
        torch.Tensor: Ein Tensor mit den CRPS-Werten, der die gleiche Form
                      wie `y_true` hat ([B, N_vars, H]). Der Loss für das Training ist
                      typischerweise `crps_values.mean()`.
    """
    # --- START OF FIX: Memory-Efficient CRPS Calculation ---
    # The original vectorized version creates a massive intermediate tensor
    # of shape [B, H, N_vars, Q], causing OOM errors with large batches.
    # This version iterates over quantiles, trading a bit of speed for
    # a massive reduction in memory usage.

    # 1. Forme y_true für die Schleife vor
    # y_true ist [B, N_vars, H]. Wir brauchen [B, H, N_vars] für die Subtraktion.
    y_true_compatible = y_true.permute(0, 2, 1)
    
    total_pinball_loss = torch.zeros_like(y_true_compatible)

    # 2. Schleife über die Quantile
    for i in range(num_quantiles):
        q = (i + 0.5) / num_quantiles
        q_tensor = torch.tensor(q, device=y_true.device)

        # Berechne die Vorhersage für dieses EINE Quantil
        # distr.icdf gibt [B, H, N_vars] zurück
        y_pred_q = distr.icdf(q_tensor)

        # Berechne den Pinball Loss für dieses Quantil
        error = y_pred_q - y_true_compatible
        pinball_loss = torch.max((q_tensor) * error, (q_tensor - 1.0) * error)
        total_pinball_loss += pinball_loss

    # 3. Approximiere den CRPS durch den Durchschnitt des Pinball Loss
    # Das Ergebnis ist [B, H, N_vars]
    crps = 2 * total_pinball_loss / num_quantiles

    # Gib den Loss in der gleichen Form wie y_true zurück: [B, N_vars, H]
    return crps.permute(0, 2, 1)
    # --- END OF FIX ---