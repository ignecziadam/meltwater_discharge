#!usr/bin/bash

# Copies DEM and LandMask tiles from source directories
# Positional arguments: {REGION [RGI_no_name] PRODTYPE [DEM/LandMask]}

# Set input and output directories
export INPUT_DIR="/scratch/atlantis2/AP_data/COP_DEM"
export OUTPUT_DIR="/scratch/atlantis2/AP_results"

# Execute the conda.sh to initialize python environments
source ~/miniconda3/etc/profile.d/conda.sh
# activate custom python environment
conda activate geotransform_env

# call python script
python ~/scripts_python/wrangle_files.py $1 $2 | \
while IFS= read -r OUT_LINE; do echo "$OUT_LINE"; done