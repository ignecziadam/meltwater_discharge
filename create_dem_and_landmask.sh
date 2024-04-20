#!usr/bin/bash

# Merging Copernicus tiles using Python Gdal installation from command line
# Positional arguments: {REGION [RGI_no_name]}

# Parse positional input arguments and set datadir
REGION=$1
DATADIR="/scratch/atlantis2/AP_results"

# Execute the conda.sh to initialize python environments
source ~/miniconda3/etc/profile.d/conda.sh
# activate custom python environment
conda activate gdal_env


#########################################################################################################
# Process the LandMask

# Navigate to the working directory (DEM or LandMask) using 'cd'
cd "${DATADIR}/${REGION}/LandMask"

echo "Preprocessing(1) LandMask for" $REGION

# list all geotiff files and save them to a file
ls ./*.tif > ./filelist.txt

# Build virtual mosaic first
gdalbuildvrt -tr 0.001 0.001 -r nearest -srcnodata 255 -vrtnodata 255 \
-input_file_list ./filelist.txt "./${REGION}_LandMask.vrt"

echo "Preprocessing(2) LandMask for" $REGION

# Modify values of the virtual mosaic [set land=1, ocean=0 (nodata==ocean)] then overwrite the original
gdal_calc.py -A "./${REGION}_LandMask.vrt" --outfile="../${REGION}_LandMask_Tmp.tif" \
--calc="0 + (A==0) + (A==2) + (A==3)" --hideNoData --NoDataValue=-1 --type=Int16 --quiet


#########################################################################################################
# Process DEMs

# Navigate to the working directory (DEM or LandMask) using 'cd'
cd "${DATADIR}/${REGION}/DEM"

echo "Preprocessing(1) DEM for ${REGION}"

# list all geotiff files and save them to a file
ls ./*.tif > ./filelist.txt

# Build virtual mosaic first
gdalbuildvrt -tr 0.001 0.001 -r bilinear -vrtnodata -9999 \
-input_file_list ./filelist.txt "./${REGION}_DEM.vrt"

echo "Preprocessing(2) DEM for ${REGION}"

# Modify values of the virtual mosaic [set land=1, ocean=0 (nodata==ocean)] then overwrite the original
gdal_calc.py -A "../${REGION}_LandMask_Tmp.tif" -B "./${REGION}_DEM.vrt" --outfile="../${REGION}_DEM_Tmp.tif" \
--calc="(A!=1)*-9999 + (A==1)*B" --NoDataValue=-9999 --quiet


########################################################################################################
# Export and reproject the modified virtual mosaics

# Get cutline path
CUTLINE="${DATADIR}/${REGION}/selectors/${REGION}_cutline_laeaa.shp"

# Navigate to the working directory (DEM or LandMask) using 'cd'
cd "${DATADIR}/${REGION}/LandMask"

echo "Exporting LandMask for ${REGION}"

gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3574 \
-cutline "${CUTLINE}" -crop_to_cutline \
-tr 250 250 -r near -srcnodata -1 -dstnodata -1 \
-of GTiff -co COMPRESS=LZW -overwrite \
"../${REGION}_LandMask_Tmp.tif" "../${REGION}_LandMask.tif"

echo "Update NoData of LandMask for ${REGION}"

gdal_calc.py -A "../${REGION}_LandMask.tif" --outfile="../${REGION}_LandMask.tif" \
--calc="(A!=1)*0 + (A==1)" --hideNoData --NoDataValue=-1 --co="COMPRESS=LZW" --quiet --overwrite

rm "../${REGION}_LandMask_Tmp.tif"

# DEM
PRODTYPE="DEM"

#
cd "${DATADIR}/${REGION}/DEM"

echo "Exporting DEM for ${REGION}"

gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3574 \
-cutline "${CUTLINE}" -crop_to_cutline \
-tr 250 250 -r bilinear -srcnodata -9999 -dstnodata -9999 \
-of GTiff -co COMPRESS=LZW -overwrite \
"../${REGION}_DEM_Tmp.tif" "../${REGION}_DEM.tif"

rm "../${REGION}_DEM_Tmp.tif"

# Finish
echo "Processing finished"