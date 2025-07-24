import torch
import optuna
import os
import time
import re
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import sys
import io
from PIL import Image
from tqdm import tqdm

# === Korrekte Imports für das neue Modell und die Utilities ===
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.utils.crps import crps_loss
from ts_benchmark.baselines.duet.utils.tools import adjust_learning_rate, EarlyStopping
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
# === NEUER IMPORT FÜR DIE FENSTER-SUCHE ===
from ts_benchmark.baselines.duet.utils.window_search import find_interesting_windows
from ...models.model_base import ModelBase

# === NEU: In-Memory-Cache für die Ergebnisse der Fenstersuche ===
# Dies verhindert, dass die teure Suche in jedem Optuna-Trial neu ausgeführt wird.
WINDOW_SEARCH_CACHE = {}

class TransformerConfig:
    """
    Konfigurationsklasse. Kombiniert Defaults mit übergebenen Argumenten.
    Bereinigt und auf die Bedürfnisse des neuen Modells zugeschnitten.
    """
    def __init__(self, **kwargs):
        defaults = {
            # --- Core Architecture ---
            "d_model": 512, "d_ff": 2048, "n_heads": 8, "e_layers": 2,
            "factor": 3, "activation": "gelu", "dropout": 0.1, "fc_dropout": 0.1,
            "output_attention": False,
            
            # --- MoE Parameters (General) ---
            "noisy_gating": True, "hidden_size": 256,
            "loss_coef": 1.0, # MoE loss coefficient
            
            # --- MoE Parameters (Expert Configuration) ---
            "num_linear_experts": 2,
            "num_esn_experts": 2,
            "esn_configs": [], # List of dicts, see `expert_factory.py`
            "k": 2,            # Default, will be overwritten below

            # --- ESN Expert Default Parameters (wird verwendet, wenn esn_configs leer ist) ---
            "reservoir_size": 256,
            "spectral_radius": 0.99,
            "sparsity": 0.1,
            "input_scaling": 1.0,
            
            # --- Training / Optimization ---
            "lr": 1e-4,
            "lradj": "cosine_warmup", "num_epochs": 100,
            "accumulation_steps": 1, # NEU: Für Gradienten-Akkumulation
            "batch_size": 128, "patience": 10,
            "num_workers": 0,  # <<< HIER HINZUFÜGEN

            # --- Data & Miscellaneous ---
            "moving_avg": 25, "CI": False, "freq": "h",
            "quantiles": [0.1, 0.5, 0.9], # Für die Inferenz
            "num_bins": 100, "tail_percentile": 0.05,
            "norm_mode": "subtract_median", # Preferred normalization mode

            # --- NEW: Projection Head Configuration ---
            "projection_head_layers": 0,       # Default to 0 for original behavior (single linear layer)
            "projection_head_dim_factor": 2,   # Hidden dim = in_features / factor
            "projection_head_dropout": 0.1,
        }

        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Abgeleitete Werte
        if hasattr(self, 'seq_len'):
            # Diese Werte werden von manchen Sub-Modulen erwartet
            setattr(self, "input_size", self.seq_len)
            setattr(self, "label_len", self.seq_len // 2) 
        else:
            raise AttributeError("Konfiguration muss 'seq_len' enthalten.")
        
        if hasattr(self, 'horizon'):
            setattr(self, "pred_len", self.horizon)
        else:
            raise AttributeError("Konfiguration muss 'horizon' enthalten.")
            
        # 'k' muss kleiner oder gleich der Gesamtanzahl Experten sein.
        # Wir setzen es hier sicherheitshalber nach der Experten-Definition.
        setattr(self, "k", min(getattr(self, "k", 1),
                           getattr(self, "num_linear_experts", 0) + getattr(self, "num_esn_experts", 0)))


class DUETProb(ModelBase):
    def __init__(self, **kwargs):
        super(DUETProb, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.seq_len = self.config.seq_len
        self.model: Optional[nn.Module] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.checkpoint_path: Optional[str] = None
        self.interesting_window_indices: Optional[Dict] = None # Für die neuen Plots

    @property
    def model_name(self) -> str:
        return "DUET-Prob-CRPS-v2"

    @staticmethod
    def required_hyper_params() -> dict:
        return {"seq_len": "input_chunk_length", "horizon": "output_chunk_length"}

    def _build_model(self):
        """
        Initialisiert das zugrundeliegende PyTorch-Modell (DUETProbModel)
        basierend auf der aktuellen Konfiguration.
        """
        if not hasattr(self.config, 'enc_in'):
             raise AttributeError("Model configuration must have 'enc_in' set before building the model. Call _tune_hyper_params() first.")
        self.model = DUETProbModel(self.config)


    def _tune_hyper_params(self, train_valid_data: pd.DataFrame):
        """
        Setzt datenabhängige Konfigurationswerte, einschließlich Frequenz, Kanalanzahl
        und Verteilungsgrenzen (channel_bounds).
        """
        # --- Frequenz-Erkennung ---
        freq = pd.infer_freq(train_valid_data.index)
        # Robustere Frequenz-Erkennung. Die ursprüngliche Logik `freq[0].lower()`
        # schlägt bei numerischen Frequenzen wie '20ms' fehl, da sie '2' zurückgibt.
        if freq is None:
            # Fallback für unregelmäßige Daten
            self.config.freq = 's'
        else:
            match = re.search(r"[a-zA-Z]", freq)
            if match:
                self.config.freq = match.group(0).lower()
            else:
                self.config.freq = 's'
        
        # --- Kanalanzahl ---
        column_num = train_valid_data.shape[1]
        self.config.enc_in = self.config.dec_in = self.config.c_out = column_num

        # --- NEU: Berechnung der Verteilungsgrenzen ---
        # Wir berechnen die Grenzen aus dem Trainingsanteil der Daten.
        # Wir nehmen hier an, dass die Aufteilung dieselbe ist wie im Haupt-Trainingslauf (90/10).
        train_data_for_bounds, _ = train_val_split(train_valid_data, 0.9, self.config.seq_len)
        channel_bounds = {}
        for col in train_data_for_bounds.columns:
            min_val, max_val = train_data_for_bounds[col].min(), train_data_for_bounds[col].max()
            buffer = 0.1 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
            channel_bounds[col] = {"lower": min_val - buffer, "upper": max_val + buffer}
        setattr(self.config, 'channel_bounds', channel_bounds)

    def _find_interesting_windows(self, valid_data: pd.DataFrame):
        """
        Sucht einmalig die "schwierigsten" Fenster im Validierungsdatensatz.
        Das Ergebnis wird in self.interesting_window_indices gespeichert.
        """
        # Der Cache-Schlüssel ist die Sequenzlänge, da die Validierungsdaten davon abhängen.
        cache_key = self.config.seq_len

        if cache_key in WINDOW_SEARCH_CACHE:
            print("--- Loading interesting windows from cache... ---")
            self.interesting_window_indices = WINDOW_SEARCH_CACHE[cache_key]
            return

        print("\n--- Searching for interesting windows for diagnostic plots (one-time search for this seq_len)... ---")
        try:
            # Die Funktion `find_interesting_windows` gibt bereits ein Dictionary
            # mit den Kanalnamen als Schlüssel zurück. Wir können es direkt verwenden.
            found_indices = find_interesting_windows(
                valid_data, self.config.horizon, self.config.seq_len
            )
            self.interesting_window_indices = found_indices

            # Speichere das Ergebnis im Cache für zukünftige Trials mit derselben seq_len
            WINDOW_SEARCH_CACHE[cache_key] = found_indices

            print("--- Found and cached interesting windows. ---\n")
        except Exception as e:
            print(f"WARNING: Could not find/cache interesting windows. Plotting will be skipped. Error: {e}")
            self.interesting_window_indices = None

    def _log_interesting_window_plots(self, epoch: int, writer: SummaryWriter, valid_dataset: Any):
        """
        Führt Inferenz auf den gefundenen "schwierigen" Fenstern durch und loggt
        die Plots in TensorBoard.
        """
        # --- NEUER FIX: Diese Plots sind für einen Horizont von 1 nicht aussagekräftig. ---
        if self.config.horizon <= 1:
            # Wir loggen diese Nachricht nur einmal pro Training, um das Terminal nicht zu überfluten.
            if not hasattr(self, '_logged_horizon_skip_warning'):
                print("\n[INFO] Diagnostic window plotting is skipped for horizon <= 1 as plots would not be meaningful.")
                self._logged_horizon_skip_warning = True
            return

        if not self.interesting_window_indices:
            return
        
        # --- FIX: Wrap the entire function in no_grad() to prevent memory leaks ---
        # This is the critical fix. Without it, every diagnostic plot creates a
        # computation graph that accumulates over epochs, causing massive slowdowns.
        with torch.no_grad():
            device = next(self.model.parameters()).device
            self.model.eval()

            for channel_name, methods in self.interesting_window_indices.items():
                for method_name, window_start_idx in methods.items():
                    # window_start_idx ist der Beginn des "Vorher"-Fensters in den rohen Validierungsdaten.
                    # Wir wollen das "Nachher"-Fenster vorhersagen, das bei `window_start_idx + horizon` beginnt.
                    forecast_start_idx = window_start_idx + self.config.horizon
                    
                    # Der Input für diese Vorhersage ist das Fenster der Länge `seq_len`, das bei `forecast_start_idx` endet.
                    # Der `forecasting_data_provider` erstellt Samples, wobei das Sample `j` dem Input `raw_data[j : j + seq_len]` entspricht.
                    # Daher ist der Index des Samples, das wir benötigen, `forecast_start_idx - seq_len`.
                    sample_idx = forecast_start_idx - self.config.seq_len

                    # Sicherheitsabfrage: Liegt der Index innerhalb der Grenzen des Datasets?
                    if not (0 <= sample_idx < len(valid_dataset)):
                        continue

                    # KORREKTUR: Greife auf das Sample über die __getitem__-Methode zu,
                    # die ein Tupel (seq_x, seq_y, seq_x_mark, seq_y_mark) zurückgibt.
                    # Die Elemente sind bereits Tensoren, nicht NumPy-Arrays.
                    input_sample_tensor, target_sample_tensor, _, _ = valid_dataset[sample_idx]

                    # Wir brauchen den Teil des Targets, der dem Horizont entspricht.
                    # Das Target aus dem Dataset enthält auch den label_len-Teil.
                    actuals_data_tensor = target_sample_tensor[-self.config.horizon:, :]

                    # Füge die Batch-Dimension hinzu und stelle sicher, dass die Daten auf dem richtigen Gerät sind.
                    input_data = input_sample_tensor.float().unsqueeze(0).to(device)
                    actuals_data = actuals_data_tensor.float().unsqueeze(0).to(device)
                    
                    denorm_distr, base_distr, _, _, _, _, _, _ = self.model(input_data)
                        
                    # === KORREKTUR: CRPS pro Kanal berechnen, nicht den globalen Durchschnitt ===
                    # crps_loss gibt einen Tensor der Form [B, N_vars, H] zurück.
                    crps_per_point = crps_loss(denorm_distr, actuals_data.permute(0, 2, 1))
                        
                    # Finde den Index des aktuellen Kanals, um den spezifischen Loss zu extrahieren.
                    try:
                        channel_names = list(self.config.channel_bounds.keys())
                        channel_idx = channel_names.index(channel_name)
                        # Berechne den mittleren CRPS für DIESEN Kanal.
                        crps_val = crps_per_point[:, channel_idx, :].mean().item()
                    except (ValueError, AttributeError):
                        # Fallback, falls der Kanal nicht gefunden wird (sollte nicht passieren).
                        # In diesem Fall wird der Plot mit dem Gesamt-CRPS beschriftet.
                        crps_val = crps_per_point.mean().item()
                        
                    # Plot erstellen
                    fig = self._create_window_plot(
                        history=input_sample_tensor.cpu().numpy(),
                        actuals=actuals_data_tensor.cpu().numpy(),
                        prediction_dist=denorm_distr,
                        channel_name=channel_name, # NEU
                        title=f'{channel_name} | {method_name} | CRPS: {crps_val:.4f}'
                    )
                        
                    # Plot in TensorBoard loggen
                    tag = f"Hard_Windows/{channel_name}/{method_name}"
                    writer.add_figure(tag, fig, global_step=epoch)
                    # WICHTIG: Schließe die Figur, um Speicherlecks zu verhindern.
                    # Ohne dies sammelt matplotlib Referenzen an, was zu massivem RAM- und Swap-Verbrauch führt.
                    plt.close(fig)

        self.model.train()

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float, trial: Optional[Any] = None) -> "ModelBase":
        self._tune_hyper_params(train_valid_data)
        config = self.config

        # Priorisiere einen existierenden log_dir aus der Konfiguration (vom Benchmark-Runner gesetzt).
        # Wenn nicht vorhanden, erstelle einen Standard-Ordner. Dies zentralisiert die Ausgabe.
        log_dir = getattr(config, 'log_dir', f'runs/{self.model_name}_{int(time.time())}')
        setattr(config, 'log_dir', log_dir) # Stelle sicher, dass er für die spätere Verwendung (z.B. Checkpoints) gesetzt ist
        writer = SummaryWriter(log_dir)
        
        # Initialisiere das Modell. `_tune_hyper_params` hat bereits alles Nötige gesetzt.
        self._build_model()

        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        print(f"--- Model Analysis ---\nTotal trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # DataParallel-Unterstützung
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # Daten-Setup: Wir brauchen keinen externen Scaler, da RevIN im Modell ist.
        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        
        # === NEU: Einmalige Suche nach interessanten Fenstern für die Plots ===
        # --- DEAKTIVIERT: Die Suche nach interessanten Fenstern ist rechenintensiv und wird hier abgeschaltet. ---
        # if valid_data is not None and not valid_data.empty:
        #     self._find_interesting_windows(valid_data)

        print("\n--- Preparing data for training... ---")
        # Der Daten-Provider erwartet timeenc=1 oder 2. Wir verwenden 1.
        # Er wird die 'date'-Spalte jetzt finden und korrekt verarbeiten.
        print("INFO: Creating training sequences. This may take a moment for large datasets...")
        train_dataset, train_data_loader = forecasting_data_provider(train_data, config, timeenc=1, batch_size=config.batch_size, shuffle=True, drop_last=True)
        print(f"INFO: Training data prepared with {len(train_dataset)} samples.")
        
        valid_data_loader = None
        if valid_data is not None and not valid_data.empty:
            print("INFO: Creating validation sequences...")
            valid_dataset, valid_data_loader = forecasting_data_provider(valid_data, config, timeenc=1, batch_size=config.batch_size, shuffle=False, drop_last=False)
            print(f"INFO: Validation data prepared with {len(valid_dataset)} samples.")
        print("--- Data preparation complete. Starting training loop. ---\n")

        optimizer = Adam(self.model.parameters(), lr=config.lr)
        
        scheduler = None
        if config.lradj == "plateau" and valid_data_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=config.patience // 2, factor=0.5)

        self.early_stopping = EarlyStopping(patience=config.patience, verbose=True)

        # --- NEU: Zeitmessung für Optuna ---
        start_time = time.time()
        last_validation_time = start_time # Initialisiere mit der Startzeit
        # Hole max_resource aus der Konfiguration.
        max_training_time = getattr(config, 'max_training_time', float('inf'))
        global_step = 0
        
        # --- NEU: Hole die Akkumulationsschritte aus der Konfiguration ---
        accumulation_steps = config.accumulation_steps

        try:
            for epoch in range(config.num_epochs):
                self.model.train()
                
                # Initialisiere Listen zum Sammeln von Metriken über alle Batches einer Epoche
                epoch_total_losses, epoch_crps_losses, epoch_importance_losses, epoch_normalized_crps_losses = [], [], [], []
                epoch_channel_losses = {name: [] for name in getattr(config, 'channel_bounds', {}).keys()} # Denormalisiert
                epoch_normalized_channel_losses = {name: [] for name in getattr(config, 'channel_bounds', {}).keys()} # Normalisiert
                epoch_gate_weights_linear, epoch_gate_weights_esn = [], []
                epoch_selection_counts = []
                epoch_p_learned_matrices = []
                epoch_p_final_matrices = []
                
                # Setze den Gradienten-Speicher vor der Epoche zurück
                optimizer.zero_grad()

                # --- VERBESSERUNG: tqdm für eine informative Fortschrittsanzeige ---
                # Wir umwickeln den DataLoader mit tqdm, um einen Fortschrittsbalken zu erhalten.
                epoch_loop = tqdm(
                    train_data_loader,
                    desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                    leave=False, # Verhindert, dass für jede Epoche eine Zeile übrig bleibt
                    file=sys.stdout # Stellt sicher, dass die Ausgabe in der Konsole landet
                )
                for i, batch in enumerate(epoch_loop):
                    global_step += 1 # Inkrementiere bei jedem Batch
                    # Der Provider gibt jetzt 4 Elemente zurück: (input, target, input_mark, target_mark).
                    # Wir brauchen nur die ersten beiden und ignorieren die Zeit-Features.
                    input_data, target, _, _ = batch
                    input_data = input_data.to(device)
                    target = target.to(device)
                    
                    # Modell-Forward-Pass. Gibt jetzt 8 Werte zurück.
                    denorm_distr, base_distr, loss_importance, batch_gate_weights_linear, batch_gate_weights_esn, batch_selection_counts, p_learned, p_final = self.model(input_data)
                    
                    # Zielhorizont für die Loss-Berechnung
                    target_horizon = target[:, -config.horizon:, :] # Shape: [B, H, V]
                    
                    # === 1. Loss-Berechnung für die Optimierung (auf normalisierter Skala) ===
                    # Der Loss für die Backpropagation wird auf der normalisierten Verteilung berechnet,
                    # um Skalenunabhängigkeit zu gewährleisten.
                    norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
                    
                    # Wende optionales Clipping auf die normalisierten Ziele an.
                    loss_clip_value = getattr(config, 'loss_target_clip', None)
                    if loss_clip_value is not None:
                        norm_target_for_loss = torch.clamp(norm_target, -loss_clip_value, loss_clip_value)
                    else:
                        norm_target_for_loss = norm_target

                    # Berechne den normalisierten CRPS. Dies ist der Haupt-Loss.
                    normalized_crps_loss = crps_loss(base_distr, norm_target_for_loss).mean()
                    total_loss = normalized_crps_loss + config.loss_coef * loss_importance
                    
                    # --- NEU: Skaliere den Loss und führe Backward-Pass aus ---
                    # Der Loss wird durch die Anzahl der Akkumulationsschritte geteilt,
                    # damit die Gradienten über die Schritte hinweg gemittelt werden.
                    scaled_loss = total_loss / accumulation_steps
                    scaled_loss.backward()

                    # --- NEU: Führe den Optimizer-Schritt nur alle `accumulation_steps` aus ---
                    if (i + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad() # Setze Gradienten nach dem Schritt zurück
                    
                    # === 2. Metriken für das Logging (ohne Gradienten) ===
                    with torch.no_grad():
                        # Denormalisierter (interpretierbarer) CRPS
                        denorm_crps_per_point = crps_loss(denorm_distr, target_horizon.permute(0, 2, 1))
                        denorm_crps_loss = denorm_crps_per_point.mean()
                        # Normalisierter CRPS (ohne Clipping, für konsistentes Logging)
                        normalized_crps_per_point = crps_loss(base_distr, norm_target)

                    # Losses für die Epochen-Statistik sammeln
                    epoch_total_losses.append(total_loss.item())
                    epoch_crps_losses.append(denorm_crps_loss.item())
                    epoch_normalized_crps_losses.append(normalized_crps_loss.item()) # HINZUGEFÜGT: Logge den normalisierten Loss
                    epoch_importance_losses.append(loss_importance.item())

                    # Sammle Experten-Metriken aus diesem Batch
                    epoch_gate_weights_linear.append(batch_gate_weights_linear)
                    epoch_gate_weights_esn.append(batch_gate_weights_esn)
                    epoch_selection_counts.append(batch_selection_counts)

                    # Sammle die Channel-Masken (detached, um Speicher zu sparen)
                    epoch_p_learned_matrices.append(p_learned.detach())
                    epoch_p_final_matrices.append(p_final.detach())
                    
                    # Kanal-spezifische Losses
                    # denorm_crps_per_point ist [B, N_vars, H]
                    denorm_loss_per_channel = denorm_crps_per_point.mean(dim=(0, 2))
                    norm_loss_per_channel = normalized_crps_per_point.mean(dim=(0, 2))
                    channel_names = self.model.module.channel_names if hasattr(self.model, 'module') else self.model.channel_names
                    for i, name in enumerate(channel_names):
                        epoch_channel_losses[name].append(denorm_loss_per_channel[i].item())
                        epoch_normalized_channel_losses[name].append(norm_loss_per_channel[i].item())

                    # Hänge die aktuellen Loss-Werte an die tqdm-Fortschrittsanzeige an.
                    epoch_loop.set_postfix(
                        loss=total_loss.item(),
                        norm_crps=normalized_crps_loss.item()
                    )

                    # --- NEU: Zeitgesteuerte Zwischen-Validierung für langes Training ---
                    validation_interval_seconds = 5 * 60 # 5 Minuten
                    current_time = time.time()
                    if (current_time - last_validation_time) > validation_interval_seconds:
                        if valid_data_loader is not None:
                            print(f"\n[INFO] Time-based validation triggered after {(current_time - start_time)/60:.1f} minutes...")
                            
                            # Führe die Validierung durch. Die Funktion ist bereits so gebaut, dass sie das Modell
                            # in den eval-Modus versetzt und danach wieder in den train-Modus.
                            interim_valid_loss_denorm, interim_valid_loss_norm = self.validate(valid_data_loader, writer, epoch, device)
                            
                            # Logge die Ergebnisse mit dem global_step für eine kontinuierliche Kurve in TensorBoard.
                            # Wir verwenden einen anderen Namen, um sie von der Epochen-End-Validierung zu unterscheiden.
                            writer.add_scalar("Loss_Interim/Validation_CRPS", interim_valid_loss_denorm, global_step)
                            writer.add_scalar("Loss_Normalized_Interim/Validation_CRPS", interim_valid_loss_norm, global_step)
                            
                            print(f"[INFO] Interim validation complete. Loss: {interim_valid_loss_denorm:.4f}")
                            
                            # Setze den Timer für die nächste Validierung zurück
                            last_validation_time = current_time
                    
                # --- Logging am Ende der Epoche ---
                # Die tqdm-Schleife wird am Ende der Epoche automatisch geschlossen und aufgeräumt.
                avg_train_total_loss = np.mean(epoch_total_losses)
                avg_train_crps_loss = np.mean(epoch_crps_losses)
                avg_train_norm_crps_loss = np.mean(epoch_normalized_crps_losses)
                avg_train_importance_loss = np.mean(epoch_importance_losses)

                writer.add_scalar("Loss/Train_Total", avg_train_total_loss, epoch)
                writer.add_scalar("Loss_Normalized/Train_CRPS", avg_train_norm_crps_loss, epoch)
                writer.add_scalar("Loss/Train_CRPS", avg_train_crps_loss, epoch)
                writer.add_scalar("Loss/Train_Importance", avg_train_importance_loss, epoch)
                for name, losses in epoch_channel_losses.items():
                    writer.add_scalar(f"Loss_per_Channel/Train_{name}", np.mean(losses), epoch)
                for name, losses in epoch_normalized_channel_losses.items():
                    writer.add_scalar(f"Loss_Normalized_per_Channel/Train_{name}", np.mean(losses), epoch)

                # --- Mittelung der Experten-Metriken über die Epoche ---
                avg_gate_weights_linear = torch.stack(epoch_gate_weights_linear).mean(dim=0) if epoch_gate_weights_linear else torch.tensor([])
                avg_gate_weights_esn = torch.stack(epoch_gate_weights_esn).mean(dim=0) if epoch_gate_weights_esn else torch.tensor([])
                if epoch_selection_counts:
                    # `stack` erzeugt einen [num_batches, num_experts] Tensor.
                    # .mean(dim=0) berechnet dann den Durchschnitt pro Experte über alle Batches.
                    avg_selection_counts = torch.stack(epoch_selection_counts, dim=0).mean(dim=0)
                else:
                    avg_selection_counts = torch.tensor([])
                
                # --- Logging der Experten-Auslastung ---
                # Hole die Expertentypen einmalig, damit sie für alle Logging-Blöcke verfügbar sind.
                model_to_log = self.model.module if hasattr(self.model, 'module') else self.model
                expert_types = [exp.__class__.__name__ for exp in model_to_log.cluster.experts]

                # Log Gating-Gewichte (nur wenn vorhanden)
                if avg_gate_weights_linear.numel() > 0 or avg_gate_weights_esn.numel() > 0:
                    linear_expert_log_idx = 0
                    esn_expert_log_idx = 0
                    for expert_idx, expert_type in enumerate(expert_types):
                        if expert_type == 'Linear_extractor':
                            if linear_expert_log_idx < len(avg_gate_weights_linear):
                                weight = avg_gate_weights_linear[linear_expert_log_idx].item()
                                writer.add_scalar(f"Expert_Gating_Weights/Linear_Expert_{linear_expert_log_idx}", weight, epoch)
                                linear_expert_log_idx += 1
                        elif expert_type == 'ReservoirExpert':
                            if esn_expert_log_idx < len(avg_gate_weights_esn):
                                weight = avg_gate_weights_esn[esn_expert_log_idx].item()
                                writer.add_scalar(f"Expert_Gating_Weights/Reservoir_Expert_{esn_expert_log_idx}", weight, epoch)
                                esn_expert_log_idx += 1


                # Log Expert Selection Counts (nur wenn vorhanden)
                if avg_selection_counts.numel() > 0:
                    expert_idx = 0
                    linear_expert_count = 0
                    esn_expert_count = 0
                    for expert_type in expert_types:
                        expert_count = linear_expert_count if expert_type == 'Linear_extractor' else esn_expert_count
                        selection_count = avg_selection_counts[expert_idx].item()
                        writer.add_scalar(f"Expert_Selection_Counts/{expert_type}_{expert_count}", selection_count, epoch)
                        if expert_type == 'Linear_extractor': linear_expert_count += 1
                        else: esn_expert_count += 1
                        expert_idx += 1

                # --- Mittelung und Logging der Channel-Masken ---
                if epoch_p_learned_matrices and epoch_p_final_matrices:
                    # Die Matrizen haben die Form [B, C, C]. Wir mitteln über die Batch-Dimension.
                    # KORREKTUR: Wir müssen über alle Batches konkatenieren und dann mitteln,
                    # um eine einzelne [C, C] Matrix für den Plot zu erhalten.
                    all_p_learned = torch.cat(epoch_p_learned_matrices, dim=0) # Shape: [Total_Samples, C, C]
                    avg_p_learned = all_p_learned.mean(dim=0) # Shape: [C, C]
                    all_p_final = torch.cat(epoch_p_final_matrices, dim=0)
                    avg_p_final = all_p_final.mean(dim=0)

                # --- Validierung ---
                if valid_data_loader is not None:
                    # GEÄNDERT: validate wird writer und epoch übergeben, um intern zu loggen
                    valid_loss_denorm, valid_loss_norm = self.validate(valid_data_loader, writer, epoch, device)
                    writer.add_scalar("Loss/Validation_CRPS", valid_loss_denorm, epoch)
                    writer.add_scalar("Loss_Normalized/Validation_CRPS", valid_loss_norm, epoch)
                    
                    # Speichere den alten besten Loss, um eine Verbesserung zu erkennen
                    old_best_loss = self.early_stopping.val_loss_min
                    
                    self.early_stopping(valid_loss_denorm, self.model.state_dict())

                    # --- NEU: Optuna Pruning-Logik, jetzt am Ende der Epoche ---
                    if trial:
                        elapsed_time_since_start = time.time() - start_time
                        # Melde den Validierungs-Loss und die vergangene Zeit an Optuna
                        trial.report(valid_loss_denorm, elapsed_time_since_start)
                        if trial.should_prune():
                            print(f"  -> Trial pruned by {trial.study.pruner.__class__.__name__} after epoch {epoch + 1}.")
                            raise optuna.exceptions.TrialPruned()

                    if self.early_stopping.val_loss_min < old_best_loss:
                        if valid_dataset and self.interesting_window_indices:
                            self._log_interesting_window_plots(epoch, writer, valid_dataset)
                        
                        # Logge die Channel-Abhängigkeiten nur, wenn sich das Modell verbessert hat.
                        if 'avg_p_learned' in locals() and 'avg_p_final' in locals():
                            # Hole den Prior vom Modell
                            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                            prior_matrix = model_ref.channel_adjacency_prior

                            # Erstelle den kombinierten 3-Panel-Plot
                            fig_dependencies = self._log_dependency_heatmaps(
                                prior_matrix=prior_matrix,
                                learned_matrix=avg_p_learned.cpu().numpy(),
                                final_matrix=avg_p_final.cpu().numpy()
                            )
                            writer.add_figure("Channel_Dependencies/Combined_View", fig_dependencies, global_step=epoch)
                            # WICHTIG: Auch diese Figur schließen, um Speicherlecks zu verhindern.
                            plt.close(fig_dependencies)

                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break
                    
                    if scheduler: scheduler.step(valid_loss_denorm)

                if config.lradj != "plateau":
                    # Silencing the learning rate update to reduce terminal clutter
                    adjust_learning_rate(optimizer, epoch + 1, config, verbose=False)
                
                # === END-OF-EPOCH OPTUNA REPORTING (REMOVED) ===
                # The trial.report() call at the end of the epoch is now redundant.
                # The new, time-based intermediate validation check is the single
                # source of truth for the pruner, which prevents warnings and is more robust.
                
                # Füge einen harten Timeout hinzu, um sicherzustellen, dass kein Trial die max_training_time überschreitet.
                if (time.time() - start_time) > max_training_time:
                        print(f"Trial timed out after {(time.time() - start_time):.2f}s (max: {max_training_time}s).")
                        break # Beendet die Epochen-Schleife
                
                writer.add_scalar("Misc/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        finally:
            # Lade den besten Zustand vom EarlyStopping und speichere ihn
            if self.early_stopping and self.early_stopping.check_point:
                self.checkpoint_path = os.path.join(config.log_dir, 'best_model.pt')
                os.makedirs(config.log_dir, exist_ok=True)
                
                checkpoint_to_save = {
                    'model_state_dict': self.early_stopping.check_point,
                    'config_dict': self.config.__dict__
                }
                torch.save(checkpoint_to_save, self.checkpoint_path)
                print(f"\n--- Finalizing run ---\n>>> Best model saved to {self.checkpoint_path} <<<")
            writer.close()
        
        # Lade das beste Modell in den Speicher, um es für die Vorhersage zu verwenden
        if self.checkpoint_path:
            self.load(self.checkpoint_path)
        return self

    def validate(self, valid_data_loader, writer: SummaryWriter, epoch: int, device: torch.device) -> tuple[float, float]:
        total_denorm_loss, total_norm_loss = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                # Der Provider gibt jetzt 4 Elemente zurück. Wir ignorieren die Zeit-Features.
                input_data, target, _, _ = batch
                input_data = input_data.to(device)
                target = target.to(device)
                
                target_horizon = target[:, -self.config.horizon:, :]

                # Unpack to match new signature (8 values)
                denorm_distr, base_distr, _, _, _, _, _, _ = self.model(input_data)
                
                # Berechne den denormalisierten Loss für EarlyStopping/Optuna
                # crps_loss erwartet target in [B, N_vars, H]
                denorm_loss = crps_loss(denorm_distr, target_horizon.permute(0, 2, 1)).mean()
                total_denorm_loss.append(denorm_loss.item())

                # Berechne den normalisierten Loss für das Logging
                norm_target = denorm_distr.normalize_value(target_horizon).permute(0, 2, 1)
                norm_loss = crps_loss(base_distr, norm_target).mean()
                total_norm_loss.append(norm_loss.item())

        self.model.train()
        return np.mean(total_denorm_loss), np.mean(total_norm_loss)

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call forecast_fit() first.")
            
        input_data = train.iloc[-self.seq_len:, :].values
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        self.model.eval()
        with torch.no_grad():
            # FIX: Unpack 8 values to match the new model signature.
            distr, base_distr, _, _, _, _, _, _ = self.model(input_tensor)
            
            # Hole Quantil-Vorhersagen
            q_list = self.config.quantiles
            q_tensor = torch.tensor(q_list, device=device, dtype=torch.float32)
            
            # distr.icdf gibt bei mehreren Quantilen [B, H, V, Q] zurück
            quantile_preds = distr.icdf(q_tensor)
        
        # Permutiere für das erwartete Output-Format [V, H, Q]
        # Wir nehmen das erste (und einzige) Element aus dem Batch
        output_array = quantile_preds.squeeze(0).permute(1, 0, 2).cpu().numpy()
        return output_array

    def load(self, checkpoint_path: str) -> "ModelBase":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Lade die Konfiguration aus dem Checkpoint, um das Modell neu zu erstellen
        config_dict = checkpoint['config_dict']
        self.config = TransformerConfig(**config_dict)
        
        self.model = DUETProbModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model successfully loaded from {checkpoint_path}.")
        return self

    def _create_window_plot(self, history, actuals, prediction_dist, channel_name, title):
        """Erstellt eine Matplotlib-Figur für ein einzelnes Fenster."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Finde den Index des zu plottenden Kanals
        try:
            # VERBESSERUNG: Hole die Kanalreihenfolge direkt vom Modell, das ist robuster.
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            channel_idx = model_ref.channel_names.index(channel_name)
        except (ValueError, AttributeError, IndexError):
             # Fallback, wenn die Namen nicht übereinstimmen
            # KORREKTUR: Tippfehler 'channel' zu 'channel_idx'
            channel_idx = 0

        # Daten vorbereiten
        horizon_len = actuals.shape[0]
        
        # === KORREKTUR: Zeige nur `horizon` Schritte der Historie an ===
        # Das entspricht dem "Vorher"-Fenster aus der `find_interesting_windows`-Suche.
        history_to_plot = history[-horizon_len:, :]
        
        # Zeitachsen-Indizes für den Plot
        history_x = np.arange(horizon_len)
        forecast_x = np.arange(horizon_len, horizon_len + horizon_len)
        
        # History und Actuals für den spezifischen Kanal
        history_y = history_to_plot[:, channel_idx]
        actuals_y = actuals[:, channel_idx]
        
        # Quantile aus der Konfiguration holen
        quantiles = self.config.quantiles
        # Das Gerät (device) aus der Verteilung holen, um den Quantil-Tensor zu erstellen
        device = prediction_dist.mean.device
        q_tensor = torch.tensor(quantiles, device=device, dtype=torch.float32)
        
        # Vorhersagen holen: [B, H, V, Q] -> [H, V, Q]
        # .icdf() gibt bei mehreren Quantilen eine zusätzliche Dimension zurück
        quantile_preds = prediction_dist.icdf(q_tensor).squeeze(0).cpu().numpy()
        
        # Vorhersagen für den spezifischen Kanal
        # KORREKTUR: Wähle den Kanal aus, bevor du die Quantile indizierst.
        # Die Form ändert sich von [H, V, Q] zu [H, Q].
        preds_y = quantile_preds[:, channel_idx, :]
        
        # Median-Index finden
        try:
            median_idx = quantiles.index(0.5)
        except (ValueError, AttributeError):
            # Fallback, falls 0.5 nicht in der Liste ist oder quantiles kein list-Objekt ist
            median_idx = len(quantiles) // 2

        # --- Plotting ---
        # History
        ax.plot(history_x, history_y, label="History", color="gray")
        
        # Vertikale Linie, die History von Forecast trennt
        ax.axvline(x=history_x[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start')

        # Actuals
        ax.plot(forecast_x, actuals_y, label="Actual", color="black", linewidth=2, zorder=10)
        
        # Confidence Intervals
        num_ci_levels = len(quantiles) // 2
        # Define a base alpha. Wider CIs (i=0) will be more transparent.
        base_alpha = 0.1
        alpha_step = 0.15

        for i in range(num_ci_levels):
            lower_q_idx = i
            upper_q_idx = len(quantiles) - 1 - i
            
            # The narrowest interval (largest i) will be the most opaque.
            current_alpha = base_alpha + (i * alpha_step)

            ax.fill_between(
                forecast_x, preds_y[:, lower_q_idx], preds_y[:, upper_q_idx],
                alpha=current_alpha, color='C0', # Use a consistent color for all intervals
                label=f"CI {quantiles[lower_q_idx]}-{quantiles[upper_q_idx]}"
            )
            
        # Median Forecast
        ax.plot(forecast_x, preds_y[:, median_idx], label="Median Forecast", color="blue", linestyle='--', zorder=11)
        
        # Layout
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # WICHTIG: Die Figur zurückgeben, damit sie geloggt werden kann
        return fig

    def _plot_single_heatmap(self, ax, matrix, title, channel_names, vmin=0, vmax=1):
        """Helper to draw a single heatmap on a given Matplotlib axis."""
        im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)

        if channel_names and len(channel_names) == matrix.shape[0]:
            ax.set_xticks(np.arange(len(channel_names)))
            ax.set_yticks(np.arange(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha="right")
            ax.set_yticklabels(channel_names)

        # Annotate cells with values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Choose text color based on background brightness for better readability
                text_color = "w" if matrix[i, j] < 0.6 * vmax else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=text_color)

        ax.set_title(title)
        return im # Return the image object for the colorbar

    def _log_dependency_heatmaps(self, prior_matrix, learned_matrix, final_matrix):
        """Creates a 3-panel matplotlib figure showing the dependency matrices."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
        fig.suptitle("Channel Dependency Analysis (Epoch Average)", fontsize=16)

        channel_names = list(getattr(self.config, 'channel_bounds', {}).keys())
        n_vars = len(channel_names)

        # --- Panel 1: User Prior ---
        if prior_matrix is not None:
            prior_np = prior_matrix.cpu().numpy()
        else:
            # If no prior is given, it's equivalent to a matrix of all ones.
            prior_np = np.ones((n_vars, n_vars))

        im1 = self._plot_single_heatmap(axes[0], prior_np, "User Prior", channel_names, vmin=0, vmax=1)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # --- Panel 2: Learned (Unconstrained) ---
        im2 = self._plot_single_heatmap(axes[1], learned_matrix, "Learned (Unconstrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # --- Panel 3: Effective (Constrained) ---
        im3 = self._plot_single_heatmap(axes[2], final_matrix, "Effective (Constrained)", channel_names, vmin=0, vmax=1)
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        return fig