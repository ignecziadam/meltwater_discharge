#!usr/bin/bash

# Merging Copernicus tiles using Python Gdal installation from command line
# Positional arguments: {REGION [RGI_no_name]}

# Parse positional input arguments adn set datasir
REGION=$1
DATADIR="/scratch/atlantis2/AP_results"

# Execute the conda.sh to initialize python environments
source ~/miniconda3/etc/profile.d/conda.sh
# activate custom python environment
conda activate gdal_env

########################################################################################################
# Process the LandMask

# Navigate to the working directory (DEM or LandMask) using 'cd'
echo "Processing ${REGION}"

cd ${DATADIR}/${REGION}/IceMask

mkdir ../temp/

# Get bounding box coordinates of the LandMask
LANDMASK="${DATADIR}/${REGION}/${REGION}_LandMask.tif"
ICEMASK=$(ls ./*.shp)

echo "${ICEMASK}"

BOX=$(gdalinfo -json ${LANDMASK} | python ~/scripts_python/get_bounding_box.py)

echo "${BOX}"


# For Greenland get PGIC (Peripheral Glacier & Ice Caps) ice mask and complete ice mask, from RGI and GIMP
if [ "${REGION}" = "RGI_5_Greenland" ]
then
	# Reproject the GIMP ice mask for Greenland
	echo "Reproject GIMP mask for Greenland"
	
	gdalwarp \
	-t_srs EPSG:3574 \
	-dstnodata -1 \
	-te $BOX -tr 250 250 -r near \
	-of GTiff -co COMPRESS=LZW -ot "Int16" -overwrite \
	"./GimpIceMask_90m_2000_v1.2.tif" "../temp/${REGION}_IceMask.tif"
	
	# Mask out the ocean
	echo "Mask out the ocean"
	
	gdal_calc.py -A "${LANDMASK}" -B "../temp/${REGION}_IceMask.tif" --outfile="../${REGION}_IceMask.tif" \
	--calc="(A==1)*(B==1) + logical_or(A!=1,B!=1)*0" --hideNoData --NoDataValue=-1 --co="COMPRESS=LZW" \
	--overwrite --quiet
	
	rm ../temp/${REGION}_IceMask.tif
	
	# Reproject the RGI vector file for Greenland (which represent PGICs)
	echo "Reproject RGI for Greenland"
	
	ogr2ogr \
	-t_srs EPSG:3574 \
	-overwrite \
	"../temp/${REGION}_IceMask_Reprojected.shp" "${ICEMASK}"
	
	# Rasterise RGI polygons for Greenland (which represent PGICs)
	echo "Rasterise RGI for Greenland PGIC"
	
	gdal_rasterize \
	-l ${REGION}_IceMask_Reprojected \
	-burn 1 -a_nodata -1 \
	-tr 250 250 -te $BOX \
	-of GTiff -co COMPRESS=LZW -ot "Int16" \
	"../temp/${REGION}_IceMask_Reprojected.shp" "../temp/${REGION}_IceMaskPGIC.tif"
	
	rm ../temp/${REGION}_IceMask_Reprojected*
	
	# Mask out the ocean
	echo "Mask out the ocean"
	
	gdal_calc.py -A "${LANDMASK}" -B "../temp/${REGION}_IceMaskPGIC.tif" --outfile="../${REGION}_IceMaskPGIC.tif" \
	--calc="(A==1)*(B==1) + logical_or(A!=1,B!=1)*0" --hideNoData --NoDataValue=-1 --co="COMPRESS=LZW" \
	--overwrite --quiet
	
	rm ../temp/${REGION}_IceMaskPGIC.tif

# Elsewhere, just use the RGI polygons to get an ice mask	
else
	# First reproject the RGI vector file
	echo "Reproject RGI for ${REGION}"
	
	ogr2ogr \
	-t_srs EPSG:3574 \
	-overwrite \
	"../temp/${REGION}_IceMask_Reprojected.shp" "${ICEMASK}"
	
	# Rasterise RGI polygons for all other regions
	echo "Rasterise RGI for ${REGION}"

	gdal_rasterize \
	-l ${REGION}_IceMask_Reprojected \
	-burn 1 -a_nodata -1 \
	-tr 250 250 -te $BOX \
	-of GTiff -co COMPRESS=LZW -ot "Int16" \
	"../temp/${REGION}_IceMask_Reprojected.shp" "../temp/${REGION}_IceMask.tif"
	
	rm ../temp/${REGION}_IceMask_Reprojected*
	
	# Mask out the ocean
	echo "Mask out the ocean"
	
	gdal_calc.py -A "${LANDMASK}" -B "../temp/${REGION}_IceMask.tif" --outfile="../${REGION}_IceMask.tif" \
	--calc="(A==1)*(B==1) + logical_or(A!=1,B!=1)*0" --hideNoData --NoDataValue=-1 --co="COMPRESS=LZW" \
	--overwrite --quiet
	
	rm ../temp/${REGION}_IceMask.tif
fi

echo "Processing finished"