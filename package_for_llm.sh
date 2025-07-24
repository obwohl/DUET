#!/bin/bash

# ==============================================================================
#             Project Packager for Large Language Model (LLM) Sessions
# ==============================================================================
# This script collects all key source files and a project structure overview
# into a single, clean directory named 'upload'. This allows for quick and
# easy context sharing in a new chat session.
#
# USAGE:
# 1. Place this script in the root directory of your project.
# 2. Make it executable: chmod +x package_for_llm.sh
# 3. Run it from the root directory: ./package_for_llm.sh
# ==============================================================================

# --- 1. Configuration ---
# The name of the directory where all files will be collected.
UPLOAD_DIR="upload"

# An array of all relevant files to be included in the package.
# Add or remove files here as the project evolves.
FILES_TO_COPY=(
    "scripts/multivariate_forecast/my_script/eisbach_proba.sh"
    "simple_inference.py"
    "ts_benchmark/baselines/duet/utils/masked_attention.py"
    "ts_benchmark/baselines/duet/duet_prob.py"
    "ts_benchmark/baselines/duet/models/duet_prob_model.py"
    "ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py"
    "ts_benchmark/baselines/duet/spliced_binned_pareto_standalone.py"
    "ts_benchmark/baselines/duet/layers/linear_pattern_extractor.py"
    "ts_benchmark/baselines/duet/utils/tools.py"
    "ts_benchmark/baselines/duet/layers/RevIN.py"
    "ts_benchmark/baselines/utils.py" 
    "ts_benchmark/baselines/duet/utils/crps.py"
    "ts_benchmark/baselines/duet/tests/test_channel_masking.py"
    "ts_benchmark/baselines/duet/tests/test_prior_impact.py"
)

# --- 2. Script Execution ---

echo "--- Preparing Project Package for LLM Session ---"

# Create a fresh upload directory, removing the old one if it exists.
echo
echo "1. Creating fresh upload directory: './${UPLOAD_DIR}'"
rm -rf "$UPLOAD_DIR"
mkdir "$UPLOAD_DIR"

# Copy all specified files into the upload directory.
echo
echo "2. Copying key project files..."
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$UPLOAD_DIR/"
        echo "   - Copied: $file"
    else
        # Add a warning if a file from the list is not found.
        echo "   - WARNING: File not found, skipped: $file"
    fi
done

# Generate the project structure tree.
echo
echo "3. Generating project structure file (tree.txt)..."
# First, check if the 'tree' command is available.
if command -v tree &> /dev/null; then
    # The 'tree' command with ignores for clutter like __pycache__, .git, etc.
    tree -I "upload|__pycache__|.git|runs|result|*.pyc|*.db*|venv|*.egg-info|dataset" > "${UPLOAD_DIR}/tree.txt"
    echo "   - 'tree.txt' created successfully."
else
    # Provide a helpful message if 'tree' is not installed.
    echo "   - WARNING: 'tree' command not found. 'tree.txt' could not be generated."
    echo "   - To install on macOS, run: brew install tree"
    echo "   - To install on Debian/Ubuntu, run: sudo apt-get install tree"
    # Create an empty file as a placeholder
    touch "${UPLOAD_DIR}/tree.txt"
fi

# Final confirmation message.
echo
echo "✅ Packaging complete."
echo "The folder './${UPLOAD_DIR}' is now ready."
echo "You can drag and drop its contents (all files + tree.txt) into a new chat window."