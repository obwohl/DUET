import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    
    A layer that normalizes a time series channel-wise and can reverse the process.
    This version is simplified to subtract the last or median value, or perform no
    normalization at all, helping to stabilize training for non-stationary series.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, norm_mode='identity'):
        """
        Args:
            num_features (int): The number of features or channels in the time series.
            eps (float): A small value added for numerical stability (to avoid division by zero).
            affine (bool): If True, this module has learnable affine parameters (gamma and beta).
            norm_mode (str): The normalization mode. One of:
                             - 'identity': No normalization is applied. The input is passed through.
                             - 'subtract_last': Subtracts the last value of the sequence from the entire sequence.
                             - 'subtract_median': Subtracts the median of the sequence from the entire sequence.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.norm_mode = norm_mode

        # Validate the normalization mode
        if self.norm_mode not in ['identity', 'subtract_last', 'subtract_median']:
            raise ValueError(f"Unknown norm_mode: '{self.norm_mode}'")

        if self.affine:
            # Learnable affine parameters (reshaped for broadcasting)
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x: torch.Tensor, mode: str, stats_to_use=None):
        """
        Performs forward normalization or backward de-normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, NumFeatures].
            mode (str): Either 'norm' for normalization or 'denorm' for de-normalization.
            stats_to_use (torch.Tensor, optional): External stats to use for de-normalization.
                                                    Shape: [Batch, NumFeatures, 2].
                                                    If provided during 'denorm', the layer will
                                                    use these instead of its internally stored stats.
        
        Returns:
            torch.Tensor: The normalized or de-normalized tensor.
            torch.Tensor (optional): If mode is 'norm', returns the statistics used for normalization.
                                     Shape: [Batch, NumFeatures, 2]
        """
        if mode == 'norm':
            self._get_statistics(x)
            x_norm = self._normalize(x)
            # The stats tensor has shape [B, N, 2] for location and scale
            stats = torch.stack([self.location_stat.squeeze(1), self.scale_stat.squeeze(1)], dim=-1)
            return x_norm, stats
            
        elif mode == 'denorm':
            x_denorm = self._denormalize(x, stats_to_use)
            return x_denorm
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def _get_statistics(self, x):
        """Calculates and stores the statistics for normalization."""
        if self.norm_mode == 'identity':
            # For identity, location is 0 and scale is 1.
            self.location_stat = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            self.scale_stat = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            
        elif self.norm_mode == 'subtract_last':
            # Location is the last value in the sequence.
            self.location_stat = x[:, -1:, :].detach()
            # Scale is the RMS of the sequence after subtracting the location.
            x_centered = x - self.location_stat
            self.scale_stat = torch.sqrt(torch.mean(x_centered**2, dim=1, keepdim=True) + self.eps).detach()

        elif self.norm_mode == 'subtract_median':
            # Location is the median value of the sequence.
            self.location_stat = torch.median(x, dim=1, keepdim=True)[0].detach()
            # Scale is the RMS of the sequence after subtracting the location.
            x_centered = x - self.location_stat
            self.scale_stat = torch.sqrt(torch.mean(x_centered**2, dim=1, keepdim=True) + self.eps).detach()
            
    def _normalize(self, x):
        """Applies the normalization."""
        x_norm = (x - self.location_stat) / self.scale_stat
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

    def _denormalize(self, x, stats_to_use):
        """
        Reverses the normalization.
        Can use either internal stats or externally provided ones.
        """
        # Determine which stats to use
        if stats_to_use is not None:
            # Reshape external stats for broadcasting: [B, N, 2] -> [B, 1, N]
            location = stats_to_use[..., 0].unsqueeze(1)
            scale = stats_to_use[..., 1].unsqueeze(1)
        else:
            location = self.location_stat
            scale = self.scale_stat
            
        x_denorm = x
        if self.affine:
            # Correctly reverse the affine transformation (symmetric operation)
            x_denorm = (x_denorm - self.beta) / self.gamma
        
        x_denorm = x_denorm * scale + location
        return x_denorm