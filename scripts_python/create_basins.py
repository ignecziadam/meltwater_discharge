# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

# Import modules
################
import os
import sys
import math
import geopandas
import xarray
import rioxarray
import numpy
import pandas
from whitebox.whitebox_tools import WhiteboxTools
from rasterio.features import shapes as rasterioshape
from shapely.geometry import shape as shapelyshape

wbt = WhiteboxTools()

# Use compression when saving GeoTiff files
wbt.set_compress_rasters(True)


# Initialise with parameters
###################################
# Set up parameters
MinBasinArea = 10  # Size of the smallest basin possible (km^2), basins below this size are merged with their larger neighbours
Resolution = 250  # Spatial resolution of the basin product (m)
EdgeNoRGI = 1  # Remove basin if >= the number of pixel touching RGI domain edges
RelBasinReduction = 10  # the % of original basin outside of the MAR domain above which to remove basin

# Set up current data directories
WorkingDIR = "C:\postdoc_bristol\gis\Regions"
CurrentDomainSTR = sys.argv[1]  # "RGI_6_Iceland" sys.argv[1]


# Set up directories and filenames
###################################
DomainDIR = os.path.join(WorkingDIR, CurrentDomainSTR)

# Set up directory for temporary files
TempDIR = os.path.join(DomainDIR, "temp")

if not os.path.lexists(TempDIR):
    os.mkdir(TempDIR)

# Create input and output filenames
InputDEM = os.path.join(DomainDIR, CurrentDomainSTR + "_DEM.tif")
InputLandMask = os.path.join(DomainDIR, CurrentDomainSTR + "_LandMask.tif")
MARDomain = os.path.join(DomainDIR, "selectors", CurrentDomainSTR + "_MARextent_laeaa.shp")
RGIDomain = os.path.join(DomainDIR, "selectors", CurrentDomainSTR + "_cutline_laeaa.shp")

Basins = os.path.join(DomainDIR, CurrentDomainSTR + "_Basins.tif")
OutflowPoints = os.path.join(DomainDIR, CurrentDomainSTR + "_OutflowPoints.shp")


# Create drainage basins
################################
# Fill the DEM
print("Fill the DEM", flush=True)

FilledDEM = os.path.join(TempDIR, CurrentDomainSTR + "_FilledDEM.tif")

wbt.fill_depressions(
    dem=InputDEM,
    output=FilledDEM,
    fix_flats=True
)

# Create the flow direction pointer
print("Calculate flow directions", flush=True)

D8Pointer = os.path.join(TempDIR, CurrentDomainSTR + "_D8Pointer.tif")

wbt.d8_pointer(
    dem=FilledDEM,
    output=D8Pointer
)

# Create the basin raster
print("Get the basins", flush=True)

BasinsRaw = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsRaw.tif")

wbt.basins(
    d8_pntr=D8Pointer,
    output=BasinsRaw
)

# Remove small basins, set them to NoData
print("Remove small basins", flush=True)

PixNo = math.ceil((MinBasinArea * 1000**2) / (Resolution**2))
BasinsFilter = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsFilter.tif")

wbt.filter_raster_features_by_area(
    i=BasinsRaw,
    output=BasinsFilter,
    threshold=PixNo,
    background="nodata"
)

# Fill small basin areas (and oceans) using nearest neighbour interpolation
print("Fill basin gaps (NN)", flush=True)

BasinsFill = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsFill.tif")

wbt.euclidean_allocation(
    i=BasinsFilter,
    output=BasinsFill,
)

# Paste the filled areas to the removed small basins (and oceans)
print("Combine filtered and filled basins", flush=True)

BasinsCombined = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsCombined.tif")

wbt.update_nodata_cells(
    input1=BasinsFilter,
    input2=BasinsFill,
    output=BasinsCombined,
)

# Assigning NoData to ocean pixels
print("Update ocean pixels", flush=True)

BasinsMasked = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsMasked.tif")

wbt.conditional_evaluation(
    i=InputLandMask,
    output=BasinsMasked,
    statement="value == 1",
    true=BasinsCombined,
    false=-1
)


# Apply MAR and RGI domains on the basins
##########################################
print("Apply MAR domain on the basins", flush=True)

BasinsClipped = os.path.join(TempDIR, CurrentDomainSTR + "_BasinsClipped.tif")

# Load the basins
basins_masked = rioxarray.open_rasterio(BasinsMasked)

basins_masked = basins_masked.squeeze(dim="band")
basins_masked = basins_masked.where(basins_masked != -1.0)
basins_masked.rio.write_nodata(-1.0, encoded=True, inplace=True)

# Load MAR and RGI domains
mar_domain = geopandas.read_file(MARDomain)
rgi_domain = geopandas.read_file(RGIDomain)

# Remove basins touching the RGI domain boundary
rgi_buffer = rgi_domain.boundary.buffer(Resolution)

rgi_edge_ID = numpy.unique(
    basins_masked.rio.clip(rgi_buffer.geometry.values, all_touched=False),
    return_counts=True
)
rgi_edge_ID = rgi_edge_ID[0][rgi_edge_ID[1] >= EdgeNoRGI]

basins_masked = xarray.where(numpy.isin(basins_masked, rgi_edge_ID), numpy.nan, basins_masked)

# Get complete basin area
basin_area_comp = numpy.unique(basins_masked, return_counts=True)

# Clip the basins with the MAR domain
basins_clip = basins_masked.rio.clip(mar_domain.geometry.values, all_touched=False)

# Get clipped basin area
basin_area_clip = numpy.unique(basins_clip, return_counts=True)

#
erase_basin_ID = []

for idx, basin_ID in enumerate(basin_area_comp[0]):

    if basin_area_clip[1][basin_area_clip[0] == basin_ID].size == 0:
        erase_basin_ID.append(basin_ID)

    else:
        full_a = basin_area_comp[1][basin_area_comp[0] == basin_ID]
        clip_a = basin_area_clip[1][basin_area_clip[0] == basin_ID]

        rel_dif = (full_a - clip_a) / full_a * 100

        if rel_dif[0] >= RelBasinReduction:
            erase_basin_ID.append(basin_ID)

basins_masked = xarray.where(numpy.isin(basins_masked, erase_basin_ID), numpy.nan, basins_masked)

# Save the basins
basins_masked.rio.write_nodata(-1.0, encoded=True, inplace=True)
basins_masked.rio.to_raster(BasinsClipped, compress='LZW')

basins_masked.close()


# Find the discharge points
############################
# Find the outflow pixels from the flow direction raster
print("Get raw outflow points", flush=True)

OutflowRaster = os.path.join(TempDIR, CurrentDomainSTR + "_OutflowRaster.tif")
OutflowPointsRaw = os.path.join(TempDIR, CurrentDomainSTR + "_OutflowPointsRaw.shp")

wbt.conditional_evaluation(
    i=D8Pointer,
    output=OutflowRaster,
    statement="value == 0",
    true=1,
    false=None
)

# Convert to these pixels to vector (points)
wbt.raster_to_vector_points(
    i=OutflowRaster,
    output=OutflowPointsRaw,
)

# Sample the filtered basin raster (i.e. small basins masked out) at the outflow points
print("Filter outflow points", flush=True)

wbt.reinitialize_attribute_table(
    i=OutflowPointsRaw
)

wbt.extract_raster_values_at_points(
    inputs=BasinsFilter,
    points=OutflowPointsRaw,
    out_text=False
)

# Only retain outflow points which do not correspond to the removed small basins
outflowpoints = geopandas.read_file(OutflowPointsRaw,
                                    ignore_fields=["FID"])

outflowpoints = outflowpoints.rename(columns={"VALUE1": "ID"})
outflowpoints["ID"] = outflowpoints["ID"].astype("int")
outflowpoints = outflowpoints[outflowpoints.ID != -32768]

# Save the filtered outflow points
outflowpoints.to_file(OutflowPoints)


# Apply MAR domain on the outflow points
#########################################
print("Apply MAR domain on the outflow points", flush=True)

wbt.reinitialize_attribute_table(
    i=OutflowPoints
)

wbt.extract_raster_values_at_points(
    inputs=BasinsClipped,
    points=OutflowPoints,
    out_text=False
)

# Only retain outflow points which do not correspond to the removed small basins
outflowpoints = geopandas.read_file(OutflowPoints,
                                    ignore_fields=["FID"])

outflowpoints = outflowpoints.rename(columns={"VALUE1": "ID"})
outflowpoints["ID"] = outflowpoints["ID"].astype("int")
outflowpoints = outflowpoints[outflowpoints.ID != -1]

# Add geographical (lat, lon) coordinates to the outputs
outflowpoints["LAT"] = outflowpoints.to_crs("EPSG:4326").geometry.y
outflowpoints["LON"] = outflowpoints.to_crs("EPSG:4326").geometry.x

# Save the filtered outflow points
outflowpoints.to_file(OutflowPoints)


# Remove basins without corresponding outflow points
#####################################################
print("Remove basins without corresponding outflow points", flush=True)

# Load the basins
basins = basins_masked
del basins_masked

# Remove basins without discharge point
basins = xarray.where(numpy.isin(basins, outflowpoints["ID"]), basins, numpy.nan)
basins.rio.write_nodata(-1.0, encoded=True, inplace=True)

# Save valid domain as shapefile
valid_mask = numpy.zeros(numpy.isfinite(basins).shape, dtype="int16")
valid_mask[numpy.isfinite(basins) == True] = 1
valid_mask_gen = rasterioshape(valid_mask, transform=basins.rio.transform())

features = []
values = []

for vec, val in valid_mask_gen:
    features.append(shapelyshape(vec))
    values.append(val)

valid_shape = geopandas.GeoDataFrame(
    pandas.DataFrame({"value": values}),
    geometry=features,
    crs=basins.rio.crs)

valid_shape = valid_shape.loc[valid_shape["value"] == 1]
valid_shape = valid_shape.dissolve(by="value")

ValidShapeFile = os.path.join(
    DomainDIR, "selectors", CurrentDomainSTR + "_Basins_extent_laeaa.shp"
)
valid_shape.to_file(ValidShapeFile)


# Save the basins
basins.rio.to_raster(Basins, compress='LZW')

basins.close()
del basins

# Clear unwanted files
#######################
os.remove(FilledDEM)
os.remove(D8Pointer)
os.remove(OutflowRaster)
os.remove(BasinsFilter)
os.remove(BasinsFill)
os.remove(BasinsCombined)
os.remove(BasinsMasked)
os.remove(BasinsClipped)

print("Finsihed processing for " + CurrentDomainSTR, flush=True)
