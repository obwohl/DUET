#!/bin/bash

# ==============================================================================
#                 DUET Probabilistic Forecasting Training Script
# ==============================================================================
# This script trains the DUET-Prob model, then AUTOMATICALLY runs the
# simple_inference.py script on the resulting best model checkpoint.
# ==============================================================================

echo "--- Starting DUET-Prob Training & Plotting Workflow ---"

echo "Clearing Python cache to ensure fresh code is used..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# --- 1. Environment Setup ---
export PYTHONPATH=/Users/friedrichsiemers/DUET:$PYTHONPATH
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# --- 2. Conda Environment Activation ---
echo "Activating Conda environment..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DUET

# --- 3. Environment Verification ---
echo "-------------------- ENVIRONMENT VERIFICATION --------------------"
echo "EXECUTING FROM: $(which python)"
echo "PYTHON VERSION: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "------------------------------------------------------------------"

# --- 4. Training Parameters ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_FILE="pressure_96.csv"
SAVE_DIR="results/eisbach/DUET-Prob-SBP_${TIMESTAMP}"
HORIZON=96
SEQ_LEN=192
EPOCHS=5
PATIENCE=5
# Add a variable for the number of plots for easy configuration
NUM_INFERENCE_PLOTS=50

echo "Starting benchmark for DUET-Prob on '${DATA_FILE}'"
echo "Horizon: ${HORIZON}, Lookback: ${SEQ_LEN}, Epochs: ${EPOCHS}"
echo "Results will be saved in: ${SAVE_DIR}"

mkdir -p "${SAVE_DIR}"
LOG_FILE="${SAVE_DIR}/training_log.txt"

# --- 5. Execute the Benchmark Script ---

python -u ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${DATA_FILE}" \
    --model-name "duet.duet_prob.DUETProb" \
    --model-hyper-params "{
        \"horizon\": ${HORIZON},
        \"seq_len\": ${SEQ_LEN},
        \"patch_len\": 48,
        \"num_epochs\": ${EPOCHS},
        \"patience\": ${PATIENCE},
        \"lr\": 0.0002472978042382883,
        \"batch_size\": 128,
        \"d_model\": 128,
        \"d_ff\": 128,
        \"n_heads\": 1,
        \"e_layers\": 1,
        \"dropout\": 0.3360569812996377,
        \"fc_dropout\": 0.0226084655215335,
        \"num_linear_experts\": 2,
        \"num_esn_experts\": 2,
        \"k\": 4,
        \"norm_mode\": \"subtract_median\",
        \"quantiles\": [0.025, 0.5, 0.975],
        \"loss_target_clip\": 5.0 
    }" \
    --num-workers 0 \
    --deterministic "full" \
    --strategy-args "{\"horizon\": ${HORIZON}, \"stride\": 1, \"save_true_pred\": false, \"save_plots\": false, \"tv_ratio\": 0.9, \"num_rollings\": 0}" \
    --timeout 60000 \
    --save-path "${SAVE_DIR}" \
    --save-true-pred False 2>&1 | tee "${LOG_FILE}"

# --- 6. Post-Training: Automatic Inference and Plotting ---
echo "--- Training finished. Extracting checkpoint path... ---"

CHECKPOINT_PATH=$(grep ">>> Best model saved to" "${LOG_FILE}" | sed 's/>>> Best model saved to \(.*\) <<</\1/' | xargs)

if [[ -n "$CHECKPOINT_PATH" && -f "$CHECKPOINT_PATH" ]]; then
    echo "Found best model at: ${CHECKPOINT_PATH}"
    echo "Pausing for 5 seconds to allow system resources to be released..."
    sleep 5
    
    # --- UPDATED CALL to the new simple_inference.py script ---
    echo "--- Running simple_inference.py ---"
    python simple_inference.py \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-file "dataset/forecasting/${DATA_FILE}" \
        --output-dir "${SAVE_DIR}/inference_plots" \
        --num-plots ${NUM_INFERENCE_PLOTS}

else
    echo "ERROR: Could not find the saved checkpoint path in the log file."
    echo "Please check '${LOG_FILE}' for errors."
    exit 1
fi

# --- 7. Completion ---
echo "✅ All benchmarks and plotting completed."
echo "The results can be found in the ${SAVE_DIR} directory."