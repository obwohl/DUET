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
    
    # 1. Erzeuge die Quantil-Niveaus
    quantiles = torch.linspace(
        start=0.5 / num_quantiles,
        end=1.0 - 0.5 / num_quantiles,
        steps=num_quantiles,
        device=y_true.device,
    )

    # 2. Berechne die vorhergesagten Werte für jedes Quantil-Niveau
    # distr.icdf gibt [B, H, N_vars, Q] zurück
    y_pred_quantiles = distr.icdf(quantiles)

    # 3. Forme y_true für das Broadcasting vor
    # y_true ist [B, N_vars, H]. Wir brauchen [B, H, N_vars, 1] für die Subtraktion.
    y_true_compatible = y_true.permute(0, 2, 1).unsqueeze(-1)

    # 4. Berechne den Pinball Loss für alle Punkte und Quantile gleichzeitig
    error = y_pred_quantiles - y_true_compatible
    
    # quantiles ist 1D [Q], wir brauchen [1, 1, 1, Q] für Broadcasting
    quantiles_bcast = quantiles.view(1, 1, 1, -1)

    loss_term1 = quantiles_bcast * error
    loss_term2 = (1.0 - quantiles_bcast) * error
    pinball_loss_per_quantile = torch.max(loss_term1, -loss_term2)

    # 5. Approximiere den CRPS durch den Durchschnitt des Pinball Loss
    # Das Ergebnis ist [B, H, N_vars]
    crps = 2 * pinball_loss_per_quantile.mean(dim=-1)
    
    # Gib den Loss in der gleichen Form wie y_true zurück: [B, N_vars, H]
    return crps.permute(0, 2, 1)