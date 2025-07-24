import torch
import torch.nn as nn

class ReservoirExpert(nn.Module):
    """
    Ein Echo State Network (ESN) Experte.

    Diese Klasse implementiert ein ESN als `nn.Module`. Es nimmt eine Zeitreihensequenz
    entgegen und gibt eine Feature-Repräsentation (den finalen Reservoir-Zustand) zurück,
    die von einer trainierbaren linearen Schicht auf `d_model` projiziert wird.

    Die internen Reservoir-Gewichte (`W_in`, `W_res`) werden zufällig initialisiert
    und NICHT trainiert. Nur die Ausgabeschicht (`readout`) ist trainierbar.
    """
    def __init__(self, config):
        super(ReservoirExpert, self).__init__()

        # Hole Hyperparameter aus der Konfiguration
        self.reservoir_size = config.reservoir_size
        self.d_model = config.d_model
        self.spectral_radius_target = config.spectral_radius
        self.sparsity = config.sparsity
        self.input_scaling = config.input_scaling

        # --- 1. Erstelle die nicht-trainierbaren Reservoir-Gewichte ---

        # Eingabegewichte: Projizieren den einzelnen Input-Wert in den Reservoir-Raum
        W_in = torch.randn(self.reservoir_size, 1) * self.input_scaling

        # Interne Reservoir-Gewichte: Definieren die komplexe Dynamik
        W_res = torch.randn(self.reservoir_size, self.reservoir_size)

        # a) Wende Sparsity an (setze zufällige Gewichte auf 0)
        num_zeros = int(self.reservoir_size * self.reservoir_size * self.sparsity)
        zero_indices = torch.randperm(self.reservoir_size * self.reservoir_size)[:num_zeros]
        W_res.view(-1)[zero_indices] = 0

        # b) Skaliere auf den Ziel-Spektralradius
        try:
            # MPS-FIX: Eigenwertberechnung auf die CPU auslagern
            eigenvalues = torch.linalg.eigvals(W_res.to('cpu'))
            current_spectral_radius = torch.max(torch.abs(eigenvalues))
            # Vermeide Division durch Null, falls die Matrix nur aus Nullen besteht
            if current_spectral_radius > 1e-9:
                # Die Skalierung erfolgt auf der Original-Matrix und ihrem Gerät
                W_res.mul_(self.spectral_radius_target / current_spectral_radius)
        except torch.linalg.LinAlgError:
            # Fallback, falls die Eigenwertberechnung fehlschlägt (sehr selten)
            print("Warnung: Eigenwertberechnung für ESN-Reservoir fehlgeschlagen. Überspringe Skalierung.")

        # Registriere die Matrizen als "buffers", damit sie mit dem Modell
        # auf das richtige Gerät verschoben werden, aber keine Gradienten bekommen.
        self.register_buffer('W_in', W_in)
        self.register_buffer('W_res', W_res)

        # --- 2. Erstelle die trainierbare Ausgabeschicht ---
        self.readout = nn.Linear(self.reservoir_size, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Führt den Forward-Pass durch das Reservoir durch.
        
        Args:
            x (torch.Tensor): Input-Tensor der Form `[Batch, SeqLen]`.
        
        Returns:
            torch.Tensor: Feature-Vektor der Form `[Batch, d_model]`.
        """
        batch_size, seq_len = x.shape
        
        # Behandelt den Fall eines leeren Batches (wichtig für SparseDispatcher)
        if batch_size == 0:
            return torch.empty(0, self.d_model, device=x.device)

        # Initialisiere den versteckten Zustand des Reservoirs für den Batch
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        # Iteriere durch die Zeitsequenz
        for t in range(seq_len):
            u_t = x[:, t].unsqueeze(1)  # Form: [Batch, 1]
            
            # ESN-Zustands-Update-Gleichung. tanh sorgt für Stabilität.
            h = torch.tanh(h @ self.W_res.T + u_t @ self.W_in.T)

        # Der finale Zustand wird durch die trainierbare Schicht projiziert
        output = self.readout(h)
        return output