#!usr/bin/bash

# Copies

# Execute the .bashrc to ammend PATH to locally installed software
source ~/miniconda3/etc/profile.d/conda.sh

# Set input and output directories
export WORKING_DIR="/scratch/atlantis2/AP_results"

# activate custom python environment (pygis) that has the necesarry modules
conda activate geotransform_env

# call python script
python ~/scripts_python/summary_results.py | \
while IFS= read -r OUT_LINE; do echo "$OUT_LINE"; done