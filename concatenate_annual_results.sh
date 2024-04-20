#!usr/bin/bash

# Copies DEM and LandMask tiles from source directories
# Positional arguments: {REGION [RGI_no_name]}

# Execute the .bashrc to ammend PATH to locally installed software
source ~/miniconda3/etc/profile.d/conda.sh

# Set input and output directories
export WORKING_DIR="/scratch/atlantis2/AP_results"

# activate custom python environment (pygis) that has the necesarry modules
conda activate geotransform_env

# call python script
python ~/scripts_python/concatenate_annual_results.py $1 | \
while IFS= read -r OUT_LINE; do echo "$OUT_LINE"; done