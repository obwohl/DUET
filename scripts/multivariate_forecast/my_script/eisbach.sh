#!/bin/bash

# HINZUGEFÜGT: Stellt sicher, dass Python die lokale, bearbeitete Version des Codes verwendet,
# anstatt einer möglicherweise installierten Version. Dies wird erreicht, indem das
# Projektverzeichnis zum PYTHONPATH hinzugefügt wird.
export PYTHONPATH=/Users/friedrichsiemers/DUET:$PYTHONPATH

# HINZUGEFÜGT: Verhindert Deadlocks bei der Verwendung von Ray und parallelen PyTorch DataLoadern auf macOS.
# Dies ist ein bekannter Workaround für Hänger, die durch verschachtelte Parallelität entstehen.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Parameter angepasst für schnelleres Stoppen und ein kleineres Modell
EPOCHS=100
PATIENCE=15

echo "Starte Benchmark für Horizont 96, loockback 288 mit shift 96_only dropout 0.1"

python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "eisbach_shifted_96_only.csv" \
    --model-name "duet.DUET" \
    --model-hyper-params "{
        \"CI\": 1,
        \"batch_size\": 512,
        \"d_ff\":32,
        \"d_model\": 32,
        \"e_layers\": 2,
        \"factor\": 3,
        \"horizon\": 96,
        \"k\": 1,
        \"loss\": \"MSE\",
        \"lr\": 0.0002,
        \"lradj\": \"cosine_warmup\",
        \"n_heads\": 2,
        \"norm\": true,
        \"num_epochs\": ${EPOCHS},
        \"num_experts\": 2,
        \"patch_len\": 48,
        \"patience\": ${PATIENCE},
        \"seq_len\": 288,
        \"num_workers\": 0,
        \"dropout\": 0.1,
        \"fc_dropout\": 0.1
    }" \
    --num-workers 0 \
    --deterministic "full" \
    --strategy-args '{"horizon": 96, "stride": 1, "save_true_pred": true}' \
    --timeout 60000 \
    --save-path "eisbach/DUET" \
    --save-true-pred True \






echo "Alle Benchmarks abgeschlossen. Die Ergebnisse sind in result/eisbach/DUET/ gespeichert."