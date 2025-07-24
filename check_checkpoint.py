import torch

# Bitte den Pfad zu Ihrer Datei anpassen
CHECKPOINT_PATH = "inference_checkpoints/duet_experiment_1751720289/best_model_checkpoint.pt" 

try:
    # Laden Sie die Datei. Wichtig: map_location='cpu' verwenden, 
    # damit es auch ohne GPU funktioniert.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # Prüfen, was geladen wurde
    if isinstance(checkpoint, dict):
        print("Die Checkpoint-Datei ist ein Wörterbuch (Dictionary).")
        print("Enthaltene Schlüssel (Keys):")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        print("\nDas ist sehr gut! Wahrscheinlich sind die Hyperparameter unter einem Schlüssel wie 'hyperparameters', 'config', 'args' oder 'model_hyperparams' gespeichert.")

    else:
        print("Die Checkpoint-Datei enthält anscheinend nur die Modellgewichte (state_dict).")
        print("In diesem Fall sind die Hyperparameter NICHT in der Datei gespeichert und müssen manuell eingetragen werden.")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")