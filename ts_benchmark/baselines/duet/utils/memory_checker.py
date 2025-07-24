import torch
import time
from typing import Dict
import numpy as np
import os
import sys

# This is a utility script, so we need to make sure it can find the other project files
try:
    from ..models.duet_prob_model import DUETProbModel
    from ..duet_prob import TransformerConfig
    from .crps import crps_loss
except ImportError:
    # If relative import fails, try adding project root to path.
    # This makes the script more robust if run from different locations.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
    from ts_benchmark.baselines.duet.duet_prob import TransformerConfig
    from ts_benchmark.baselines.duet.utils.crps import crps_loss


def find_optimal_hardware_config(
    config_dict: Dict, 
    num_channels: int, 
    num_train_samples: int, 
    max_epoch_seconds: float = 300.0
) -> tuple[int | None, int | None]:
    """
    Findet die optimale Hardware-Konfiguration (physische Batch-Größe und Akkumulationsschritte)
    für eine gegebene effektive Batch-Größe. Testet verschiedene "K-Faktoren", um die
    schnellste Epochenzeit zu finden, ohne das Speicherlimit zu überschreiten.

    Args:
        config_dict: Ein Dictionary mit den Hyperparametern für das Modell.
                     Muss 'batch_size' enthalten, das als effektive Batch-Größe dient.
        num_channels: Die Anzahl der Input-Kanäle der Daten.
        num_train_samples: Die Gesamtzahl der Samples im Trainingsdatensatz.
        max_epoch_seconds: Die maximal akzeptable geschätzte Zeit für eine Epoche in Sekunden.

    Returns:
        Ein Tupel (optimale_physische_batch_grösse, optimale_akkumulationsschritte).
        Gibt (None, None) zurück, wenn keine Konfiguration das Zeitlimit einhält.
    """
    if not torch.backends.mps.is_available():
        print("  [HW-Check] Not on MPS device. Skipping check. Using k=1.")
        # Gib die effektive Batch-Größe als physische Größe und k=1 zurück.
        return config_dict['batch_size'], 1

    device = torch.device("mps")
    
    effective_batch_size = config_dict['batch_size']

    config_dict['enc_in'] = num_channels
    config_dict['dec_in'] = num_channels
    config_dict['c_out'] = num_channels
    config_dict['channel_bounds'] = {f'var_{i}': {} for i in range(num_channels)}

    print(f"\n--- [HW-Check] Suche nach optimaler HW-Konfiguration für effektive Batch-Größe: {effective_batch_size} ---")

    K_FACTORS_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    performance_results = []

    for k in K_FACTORS_TO_TEST:
        physical_batch_size = effective_batch_size * k
        print(f"  -> Teste k={k} (Physische Batch-Größe: {physical_batch_size})...")

        try:
            # Erstelle das Modell und die Daten für diesen spezifischen Test
            temp_config = TransformerConfig(**{**config_dict, 'batch_size': physical_batch_size})
            model = DUETProbModel(temp_config).to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            dummy_input = torch.randn(physical_batch_size, temp_config.seq_len, num_channels, device=device)

            # --- FIX: Führe 5 (statt 3) Messungen durch für eine stabilere Median-Messung ---
            pass_durations = []
            for _ in range(5):
                start_time = time.time()
                optimizer.zero_grad()

                # --- FIX: Repliziere die VOLLSTÄNDIGE Loss-Berechnung aus dem Training ---
                denorm_distr, base_distr, loss_importance, _, _, _, _, _ = model(dummy_input)

                # Erstelle einen Dummy-Ziel-Tensor mit der korrekten Form [B, H, V]
                dummy_target_horizon = torch.randn(
                    physical_batch_size, temp_config.horizon, num_channels, device=device
                )

                # Normalisiere das Dummy-Ziel, wie es im Training geschieht
                norm_target = denorm_distr.normalize_value(dummy_target_horizon).permute(0, 2, 1)

                # Berechne den vollständigen Loss
                normalized_crps_loss = crps_loss(base_distr, norm_target).mean()
                total_loss = normalized_crps_loss + temp_config.loss_coef * loss_importance

                total_loss.backward()
                optimizer.step()
                pass_durations.append(time.time() - start_time)

            median_duration = np.median(pass_durations)

            # Berechne die geschätzte Epochenzeit
            # Wichtig: Die Anzahl der Schleifendurchläufe hängt von der physischen Batch-Größe ab
            num_physical_passes_per_epoch = num_train_samples // physical_batch_size
            if num_physical_passes_per_epoch == 0:
                print("     WARNUNG: Datensatz kleiner als physische Batch-Größe. Schätzung unzuverlässig.")
                # Setze eine hohe Strafe, um diesen Fall zu vermeiden
                estimated_epoch_duration = float('inf')
            else:
                estimated_epoch_duration = median_duration * num_physical_passes_per_epoch

            print(f"     Median Pass-Dauer: {median_duration:.4f}s. Geschätzte Epochenzeit: {estimated_epoch_duration:.1f}s.")
            performance_results.append({
                "k": k,
                "physical_bs": physical_batch_size,
                "estimated_time": estimated_epoch_duration
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "failed to allocate" in str(e).lower():
                print(f"     ❌ FAILED. Out of Memory bei k={k}. Breche Suche für größere k ab.")
                break # Größere k-Werte werden ebenfalls fehlschlagen
            raise e
        finally:
            torch.mps.empty_cache()

    if not performance_results:
        print("--- [HW-Check] ❌ FAILED. Keine Konfiguration konnte ausgeführt werden.")
        return None, None

    # Finde die beste Konfiguration
    best_config = min(performance_results, key=lambda x: x['estimated_time'])

    print(f"--- [HW-Check] Beste Konfiguration: k={best_config['k']} (physische BS: {best_config['physical_bs']}) mit geschätzter Epochenzeit von {best_config['estimated_time']:.1f}s.")

    # Finale Prüfung gegen das harte Limit
    if best_config['estimated_time'] > max_epoch_seconds:
        print(f"--- [HW-Check] ❌ FAILED. Selbst die schnellste Konfiguration überschreitet das Limit von {max_epoch_seconds}s.")
        return None, None

    print("--- [HW-Check] ✅ PASSED. Beste Konfiguration wird verwendet.")
    return best_config['physical_bs'], best_config['k']