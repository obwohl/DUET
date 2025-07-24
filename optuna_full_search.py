# optuna_run_heuristic_search.py
import optuna
import os
import subprocess
import json
import logging
import numpy as np
import sys
import torch
import time
import psutil 
from optuna.pruners import HyperbandPruner
import pandas as pd

# Direkter Import der notwendigen Klassen
from ts_benchmark.baselines.duet.duet_prob import DUETProb
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split


# --- 1. Logging-Konfiguration ---
logging.getLogger("optuna").setLevel(logging.INFO)
# --- WICHTIG: Ändere den Namen, um die Ergebnisse dieses Experiments zu isolieren ---
STUDY_NAME = "abfluss" 
STORAGE_NAME = "sqlite:///optuna_study.db" 

# --- 2. Feste Trainingsparameter für die lange, intensive Suche ---
FIXED_PARAMS = {
    "data_file": "abfluss.csv", 
    "horizon": 96,
    "num_epochs": 1000,
    "patience": 5,        
    "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    "lradj": "constant",
    "min_training_time": , 
    "max_training_time": 10800,
    "channel_adjacency_prior": [
        [1, 1, 0, 0, 0, 0],  
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1], 
    ]
}

def get_suggested_params(trial: optuna.Trial) -> dict:
    """Schlägt einen Satz von Hyperparametern vor."""
    params = {}
    params["seq_len"] = trial.suggest_categorical("seq_len", [48, 96, 192, 384])
    params["norm_mode"] = trial.suggest_categorical("norm_mode", ["subtract_last", "subtract_median"])
    params["lr"] = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    params["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512])
    params["d_ff"] = trial.suggest_categorical("d_ff", [32, 64, 128, 256, 512])
    params["e_layers"] = trial.suggest_int("e_layers", 1, 3)

    # --- KORREKTUR: Statischer Suchraum für n_heads ---
    # 1. Schlage n_heads immer aus der vollen Liste vor, um den Suchraum statisch zu halten.
    params["n_heads"] = trial.suggest_categorical("n_heads", [1, 2, 4, 8])

    # 2. Prüfe die Gültigkeit der Kombination und prune den Trial, wenn sie ungültig ist.
    if params["d_model"] % params["n_heads"] != 0:
        raise optuna.exceptions.TrialPruned(f"d_model ({params['d_model']}) is not divisible by n_heads ({params['n_heads']}).")
        
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    params["fc_dropout"] = trial.suggest_float("fc_dropout", 0.0, 0.5)
    
    # Optuna schlägt die exakte Batch-Größe vor, die verwendet werden soll.
    params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    params["loss_coef"] = trial.suggest_float("loss_coef", 0.1, 2.0, log=True)
    params["num_linear_experts"] = trial.suggest_int("num_linear_experts", 0, 8)
    
    if params["num_linear_experts"] == 0:
        params["num_esn_experts"] = trial.suggest_int("num_esn_experts", 1, 8)
    else:
        params["num_esn_experts"] = trial.suggest_int("num_esn_experts", 0, 8)

    total_experts = params["num_linear_experts"] + params["num_esn_experts"]
    if total_experts == 0: 
        raise optuna.exceptions.TrialPruned("Total number of experts is zero.")
    params["k"] = trial.suggest_int("k", 1, total_experts)

    params["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])

    if params["num_esn_experts"] > 0:
        params["reservoir_size"] = trial.suggest_categorical("reservoir_size", [16, 32, 64, 128, 256])
        params["spectral_radius"] = trial.suggest_float("spectral_radius", 0.6, 1.4)
        params["sparsity"] = trial.suggest_float("sparsity", 0.01, 0.5)

    params["projection_head_layers"] = trial.suggest_int("projection_head_layers", 0, 4)
    if params["projection_head_layers"] > 0:
        params["projection_head_dim_factor"] = trial.suggest_categorical("projection_head_dim_factor", [1, 2, 4, 8])
        params["projection_head_dropout"] = trial.suggest_float("projection_head_dropout", 0.0, 0.5)

    params["loss_target_clip"] = trial.suggest_categorical("loss_target_clip", [None, 5.0, 10.0, 15.0])

    return params

def run_full_hardness_check_and_prune(
    model_hyper_params: dict,
    data: pd.DataFrame
):
    """
    Führt einen Pre-Flight-Check mit voller Härte durch, indem das exakte
    Speicher-Szenario eines echten Laufs simuliert wird.
    1. Beide Datasets (train/valid) werden vorab erstellt.
    2. Es werden sowohl Trainings- (forward+backward) als auch Validierungsschritte (forward)
       simuliert.
    Bei einem OOM-Fehler wird der Trial direkt gepruned.
    """
    batch_size = model_hyper_params['batch_size']
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Full Hardness Check: Testing feasibility for batch_size = {batch_size} on {device} ---")

    # Konfiguration für den Mini-Benchmark
    N_TRAIN_TEST_STEPS = 5 # Simuliere einige Trainingsschritte
    N_VALID_TEST_STEPS = 3 # Simuliere einige Validierungsschritte
    
    model = None
    optimizer = None
    train_loader = None
    valid_loader = None

    try:
        # 1. Erstelle das Modell, um eine gültige Konfiguration für die Daten zu erhalten
        model = DUETProb(**model_hyper_params)
        model._tune_hyper_params(data)

        # 2. **VOLLE HÄRTE**: Erstelle Trainings- UND Validierungs-Dataset vor dem Test,
        # um den RAM-Bedarf des realen Laufs exakt zu simulieren.
        print("    Pre-creating training AND validation datasets to accurately simulate memory usage...")
        train_data_split, valid_data_split = train_val_split(data, 0.9, model_hyper_params['seq_len'])
        
        train_dataset, _ = forecasting_data_provider(
            train_data_split, model.config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )
        
        # Sicherstellen, dass genügend Daten für den Test vorhanden sind
        if len(train_dataset) < batch_size:
            print(f"    -> PRUNING: Not enough training data ({len(train_dataset)} samples) for batch size {batch_size}.")
            raise optuna.exceptions.TrialPruned("Not enough training data for the batch size.")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        valid_loader = None
        if valid_data_split is not None and not valid_data_split.empty:
            valid_dataset, _ = forecasting_data_provider(
                valid_data_split, model.config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
            )
            if len(valid_dataset) >= batch_size:
                 valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True
                )

        # 3. Baue das Modell, verschiebe es auf das Device und erstelle den Optimizer
        model._build_model()
        model.model.to(device)
        optimizer = torch.optim.Adam(model.model.parameters())
        print("    Model and data loaded. Starting simulated training/validation steps...")

        # 4. **VOLLE HÄRTE**: Simuliere Trainingsschritte (mit Gradientenberechnung)
        train_loader_iter = iter(train_loader)
        for i in range(min(N_TRAIN_TEST_STEPS, len(train_loader))):
            input_data, _, _, _ = next(train_loader_iter)
            input_data = input_data.to(device)
            optimizer.zero_grad()
            _, _, loss, _, _, _, _, _ = model.model(input_data)
            loss.backward()
            optimizer.step()
        print(f"    -> {min(N_TRAIN_TEST_STEPS, len(train_loader))} training steps successful.")

        # 5. **VOLLE HÄRTE**: Simuliere Validierungsschritte (ohne Gradienten)
        if valid_loader:
            valid_loader_iter = iter(valid_loader)
            with torch.no_grad():
                for i in range(min(N_VALID_TEST_STEPS, len(valid_loader))):
                    input_data, _, _, _ = next(valid_loader_iter)
                    input_data = input_data.to(device)
                    model.model(input_data) # Nur Forward-Pass
            print(f"    -> {min(N_VALID_TEST_STEPS, len(valid_loader))} validation steps successful.")
        
        # Synchronisiere, um sicherzustellen, dass alle GPU-Operationen abgeschlossen sind
        if torch.backends.mps.is_available(): torch.mps.synchronize()
        elif torch.cuda.is_available(): torch.cuda.synchronize()

        # 6. Wenn alles gut geht, ist der Test bestanden
        print("    -> SUCCESS: Full hardness check passed. Configuration is feasible.")

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
            print(f"    -> PRUNING: Out of Memory with batch_size = {batch_size} during hardness check.")
            raise optuna.exceptions.TrialPruned(f"OOM with batch_size {batch_size}")
        else:
            print(f"    -> PRUNING: Encountered a non-OOM runtime error during hardness check: {e}")
            raise optuna.exceptions.TrialPruned(f"Runtime error during hardness check: {e}")
    finally:
        # Speicher aufräumen
        del model, optimizer, train_loader, valid_loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()


def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    """Führt einen Trainingslauf durch und gibt den Validierungs-Loss zurück."""
    trial_num = trial.number
    print(f"\n\n{'='*20} STARTING TRIAL #{trial_num} {'='*20}")
    
    suggested_params = get_suggested_params(trial)
    model_hyper_params = {**FIXED_PARAMS, **suggested_params}

    print("Testing Parameters:")
    for key, value in model_hyper_params.items():
        print(f"  - {key}: {value}")

    # --- "Go/No-Go"-Check mit voller Härte ---
    # Diese Funktion wirft eine Pruned-Exception, wenn die Konfiguration nicht in den Speicher passt.
    # Dadurch wird die `objective`-Funktion hier direkt beendet.
    run_full_hardness_check_and_prune(
        model_hyper_params=model_hyper_params,
        data=data
    )
    
    save_dir = f"results/optuna_heuristic/{STUDY_NAME}/trial_{trial_num}"
    os.makedirs(save_dir, exist_ok=True)
    model_hyper_params['log_dir'] = save_dir

    try:
        # 1. Initialisiere das Modell mit den geprüften Parametern
        model = DUETProb(**model_hyper_params)

        # 2. Führe das Training aus.
        model.forecast_fit(data, train_ratio_in_tv=0.9, trial=trial)

        # 3. Extrahiere Metadaten nach dem Training
        total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        trial.set_user_attr("total_trainable_parameters", total_params)

        # 4. Gib den besten Validierungs-Loss zurück
        best_valid_loss = model.early_stopping.val_loss_min
        if not np.isfinite(best_valid_loss) or best_valid_loss is None:
             print(f"TRIAL #{trial_num} resulted in an invalid loss ({best_valid_loss}). Pruning.")
             raise optuna.exceptions.TrialPruned("Training did not produce a valid finite loss.")

        print(f"TRIAL #{trial_num} COMPLETED. Best validation loss: {best_valid_loss:.6f}, Total Params: {total_params:,}")
        return best_valid_loss

    except optuna.exceptions.TrialPruned:
        print(f"TRIAL #{trial_num} was pruned by Optuna during training.")
        raise
    except Exception as e:
        import traceback
        print(f"TRIAL #{trial_num} FAILED with an unexpected exception during the main training run.")
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned(f"Full training run failed: {e}")

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        load_if_exists=True,
        pruner=HyperbandPruner(
            min_resource=FIXED_PARAMS["min_training_time"],
            max_resource=FIXED_PARAMS["max_training_time"],
            reduction_factor=3,
        )
    )

    initial_params_B = {
            "seq_len": 48,
            "norm_mode": "subtract_median",
            "lr": 0.0003,
            "d_model": 32,
            "d_ff": 32,
            "e_layers": 1,
            "n_heads": 1,
            "dropout": 0.1,
            "fc_dropout": 0.1,
            "batch_size": 512,
            "loss_coef": 0.9,
            "num_linear_experts": 3,
            "num_esn_experts": 3,
            "k": 3,
            "hidden_size": 64,
            "reservoir_size": 64,
            "spectral_radius": 1.119676104198816,
            "sparsity": 0.24647325574607348,
            "projection_head_layers": 2,
            "projection_head_dim_factor": 2,
            "projection_head_dropout": 0.06082413522212075,
            "loss_target_clip": None
        }

    if len(study.get_trials(deepcopy=False)) == 0:
        print("Enqueuing initial trial B to warm-start the study...")
        study.enqueue_trial(initial_params_B, skip_if_exists=True)

    print(f"\nLoading data from '{FIXED_PARAMS['data_file']}' once before starting the study...")
    data_source = LocalForecastingDataSource()
    data = data_source._load_series(FIXED_PARAMS['data_file'])
    print("Data loaded successfully. Starting optimization...")

    study.optimize(lambda trial: objective(trial, data), n_trials=100)

    print("\n\n" + "="*50 + "\nHEURISTIC SEARCH FINISHED\n" + "="*50)
    try:
        print(f"Best trial: #{study.best_trial.number}")
        print(f"  Value (min valid CRPS): {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
        print(f"  User Attributes: {study.best_trial.user_attrs}")
    except ValueError:
        print("No successful trials were completed.")
    print("\nTo analyze the results, use the 'analyse_study.py' script.")