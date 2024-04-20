# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

# Import modules
################
import os
import sys
import glob
import xarray
import rioxarray
import rasterio
import numpy
import geopandas
import pandas
import multiprocessing
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterioshape
from shapely.geometry import shape as shapelyshape


# Initialise script
####################
# Get input and output directory paths from the bash shell
InputDIR = os.getenv("INPUT_DIR")  # os.getenv("INPUT_DIR") "C:\postdoc_bristol\gis\data_test\mar"
OutputDIR = os.getenv("OUTPUT_DIR")  # os.getenv("OUTPUT_DIR") "C:\postdoc_bristol\gis\Regions"

# Get positional arguments from the bash shell
CurrentDomainSTR = sys.argv[1]  # sys.argv[1] "RGI_6_Iceland"

# Set the number of cores to use in parallel processing
core_no = sys.argv[2]  # sys.argv[2] 1
core_no = int(core_no)

# Check if output directories exists, if not create them
MAROutDIR = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing")

MAROutDIR_IceRunoff = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "IceRunoffAnnual")
MAROutDIR_LandRunoff = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "LandRunoffAnnual")
MAROutDIR_IceAlbedo = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "IceAlbedoAnnual")

MAROutDIR_IceRunoffGrad = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "IceRunoffGradAnnual")
MAROutDIR_LandRunoffGrad = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "LandRunoffGradAnnual")
MAROutDIR_IceAlbedoGrad = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "IceAlbedoGradAnnual")

MAROutDIR_MARDaily = os.path.join(OutputDIR, CurrentDomainSTR, "MAR_Preprocessing", "MAR_Daily")

if not os.path.lexists(MAROutDIR):
    os.mkdir(MAROutDIR)

if not os.path.lexists(MAROutDIR_IceRunoff):
    os.mkdir(MAROutDIR_IceRunoff)
if not os.path.lexists(MAROutDIR_LandRunoff):
    os.mkdir(MAROutDIR_LandRunoff)
if not os.path.lexists(MAROutDIR_IceAlbedo):
    os.mkdir(MAROutDIR_IceAlbedo)
if not os.path.lexists(MAROutDIR_IceRunoffGrad):
    os.mkdir(MAROutDIR_IceRunoffGrad)
if not os.path.lexists(MAROutDIR_LandRunoffGrad):
    os.mkdir(MAROutDIR_LandRunoffGrad)
if not os.path.lexists(MAROutDIR_IceAlbedoGrad):
    os.mkdir(MAROutDIR_IceAlbedoGrad)
if not os.path.lexists(MAROutDIR_MARDaily):
    os.mkdir(MAROutDIR_MARDaily)

# Set up directory dictionary
dir_dict = {
    "InputDIR": InputDIR,
    "OutputDIR": OutputDIR,
    "CurrentDomainSTR": CurrentDomainSTR,
    "MAROutDIR": MAROutDIR,
    "MAROutDIR_IceRunoff": MAROutDIR_IceRunoff,
    "MAROutDIR_LandRunoff": MAROutDIR_LandRunoff,
    "MAROutDIR_IceAlbedo": MAROutDIR_IceAlbedo,
    "MAROutDIR_IceRunoffGrad": MAROutDIR_IceRunoffGrad,
    "MAROutDIR_LandRunoffGrad": MAROutDIR_LandRunoffGrad,
    "MAROutDIR_IceAlbedoGrad": MAROutDIR_IceAlbedoGrad,
    "MAROutDIR_MARDaily": MAROutDIR_MARDaily
}

# Echo the region name to logfile
print("Processing " + CurrentDomainSTR + " using MAR data from " + CurrentDomainSTR + " domain", flush=True)


# Iterate through the file list
#################################
def process_annual_mar_data(itno, file, paths):

    # Define NoData, domain name and projection dictionaries
    NoData_MAR = 9.969209968386869e+36

    # Define output coordinate system and resolution
    OutCRS = "EPSG:3574"
    OutRes = (6000, 6000)

    # Define parameters for the calculation of the vertical gradients (runoff, albedo)
    MinVertDiff = 50  # Below (<) this 8-N elevation gradient threshold, vertical gradients are assumed zero
    MinNeighNo = 5  # Only obtain vertical gradients if this number (>=) of valid values are available in the 8-N

    # Set up domain directories nto help looking for correct files (from local --> MAR)
    DomainDict = {"RGI_3_CanadaN": "Canadian_Arctic",
                  "RGI_4_CanadaS": "Canadian_Arctic",
                  "RGI_5_Greenland": "Greenland",
                  "RGI_6_Iceland": "Iceland",
                  "RGI_7_Svalbard": "Russian_Arctic",
                  "RGI_9_RussiaN": "Russian_Arctic"}

    CRSDict = {"RGI_3_CanadaN": "+proj=stere +lat_0=70 +lon_0=-70 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
               "RGI_4_CanadaS": "+proj=stere +lat_0=70 +lon_0=-70 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
               "RGI_5_Greenland": "+proj=stere +lat_0=70.5 +lon_0=-40 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
               "RGI_6_Iceland": "+proj=stere +lat_0=65 +lon_0=-19 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
               "RGI_7_Svalbard": "+proj=stere +lat_0=78 +lon_0=55 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
               "RGI_9_RussiaN": "+proj=stere +lat_0=78 +lon_0=55 +k=1 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"}

    # Get current year from filename
    CurrentYear = file.split("-")[-1].split(".")[0]
    print("Year: " + CurrentYear + " start processing", flush=True)

    # Open the actual dataset
    MAR_in = xarray.open_dataset(file, decode_coords="all")


    # Handle coordinates (x, y, time) and CRS
    ###########################################
    # Rename x, y dims (some methods only accept input if coord axes are named: 'x' 'y' appropriately)
    a = dict(MAR_in.dims)
    b = list(a.keys())
    b[0] = "x"
    b[1] = "y"

    NameMod = dict(zip(list(a.keys())[0:2], b[0:2]))
    MAR_in = MAR_in.rename(NameMod)

    # Also rename time dim
    MAR_in = MAR_in.rename({"TIME": "time"})

    # Convert from x, y coordinates in km to metres
    MAR_in = MAR_in.assign_coords(x=MAR_in["x"] * 1000)
    MAR_in = MAR_in.assign_coords(y=MAR_in["y"] * 1000)

    # Write the CRS to the dataset (using the projection dictionary)
    MAR_in = MAR_in.rio.write_crs(CRSDict[paths["CurrentDomainSTR"]])


    # SPECIAL CASE (Canada): rotate the dataset
    ############################################
    if DomainDict[paths["CurrentDomainSTR"]] == "Canadian_Arctic":
        A_rot = MAR_in.rio.transform() * Affine.rotation(10, (78.5, 153.5))
        MAR_in.rio.write_transform(A_rot, inplace=True)


    # Select, subset and handle NaNs
    #################################
    # Ice mask (using fractional %)
    Mask = MAR_in["MSK"]
    Mask.name = "Mask"
    Mask = Mask.where(Mask > 0.001)
    Mask = Mask.fillna(0)
    Mask = Mask.where(Mask < 99.999)
    Mask = Mask.fillna(100)
    Mask.rio.write_nodata(-999999, encoded=True, inplace=True)

    # Model elevation
    Height = MAR_in["SH"]
    Height.name = "Height"
    Height.rio.write_nodata(-999999, encoded=True, inplace=True)

    # Runoff on ice
    RU_ice = MAR_in["RU"].sel(SECTOR=1)
    RU_ice.name = "RU_ice"
    RU_ice = RU_ice.where(RU_ice != NoData_MAR)
    RU_ice = RU_ice.where(Mask != 0)
    RU_ice.rio.write_nodata(-999999, encoded=True, inplace=True)

    # Runoff on land
    RU_land = MAR_in["RU"].sel(SECTOR=2)
    RU_land.name = "RU_land"
    RU_land = RU_land.where(RU_land != NoData_MAR)
    RU_land = RU_land.where(MAR_in["SRF"] != 1)
    RU_land = RU_land.where(Mask != 100)
    RU_land.rio.write_nodata(-999999, encoded=True, inplace=True)

    # Surface albedo of ice
    AL_ice = MAR_in["AL2"].sel(SECTOR1_1=1)
    AL_ice.name = "AL_ice"
    AL_ice = AL_ice.where(Mask != 0)
    AL_ice.rio.write_nodata(-999999, encoded=True, inplace=True)


    # Reproject
    ##########
    # Ice mask
    Mask = Mask.rio.reproject(OutCRS, resolution=OutRes,
                              resampling=Resampling.bilinear)

    # Model elevation
    Height = Height.rio.reproject(OutCRS, resolution=OutRes,
                                  resampling=Resampling.bilinear)

    # Runoff on ice
    RU_ice = RU_ice.rio.reproject(OutCRS, resolution=OutRes,
                                  resampling=Resampling.bilinear)

    # Runoff on land
    RU_land = RU_land.rio.reproject(OutCRS, resolution=OutRes,
                                    resampling=Resampling.bilinear)

    # Surface albedo of ice
    AL_ice = AL_ice.rio.reproject(OutCRS, resolution=OutRes,
                                  resampling=Resampling.bilinear)


    # Clip the 2D and 3D data arrays with the RGI boundaries
    #########################################################
    # Import RGI boundaries for clipping
    ClipFile = os.path.join(paths["OutputDIR"], paths["CurrentDomainSTR"], "selectors", paths["CurrentDomainSTR"] + "_cutline_laeaa.shp")
    ClipShape = geopandas.read_file(ClipFile)

    # Execute clip
    Mask = Mask.rio.clip(ClipShape.geometry.values, all_touched=False)
    Height = Height.rio.clip(ClipShape.geometry.values, all_touched=False)
    RU_ice = RU_ice.rio.clip(ClipShape.geometry.values, all_touched=False)
    RU_land = RU_land.rio.clip(ClipShape.geometry.values, all_touched=False)
    AL_ice = AL_ice.rio.clip(ClipShape.geometry.values, all_touched=False)


    # Get a polygon for the valid areas (raw MAR coverage)
    ###############################################################
    if itno == 0:
        ValidMask = numpy.zeros(numpy.isfinite(Mask).shape, dtype="int16")
        ValidMask[numpy.isfinite(Mask) == True] = 1
        ValidMaskGen = rasterioshape(ValidMask, transform=Mask.rio.transform())

        Features = []
        Values = []

        for vec, val in ValidMaskGen:
            Features.append(shapelyshape(vec))
            Values.append(val)

        ValidShape = geopandas.GeoDataFrame(
            pandas.DataFrame({"value": Values}),
            geometry=Features,
            crs=OutCRS)

        ValidShape = ValidShape.loc[ValidShape["value"] == 1]
        ValidShape = ValidShape.dissolve(by="value")

        ValidShapeFile = os.path.join(
            paths["OutputDIR"], paths["CurrentDomainSTR"], "selectors", paths["CurrentDomainSTR"] + "_MARextent_laeaa.shp"
        )
        ValidShape.to_file(ValidShapeFile)


    # Define custom function to calculate 8-N vertical gradients in an array
    #########################################################################
    def vertgrad8n(arr, elev, min_elev_diff, min_ne_no):
        # Create rolling window views of the variable (e.g. runoff) and elevation
        a_ro = arr.rolling({"y": 3, "x": 3}, center=True).construct({"y": "ry", "x": "rx"})
        e_ro = elev.rolling({"y": 3, "x": 3}, center=True).construct({"y": "ry", "x": "rx"})

        # Calculate centralised 8-N differences for both
        a_cdiff = a_ro[:, :, :, 1, 1] - a_ro[:, :, :, :, :]
        e_cdiff = e_ro[:, :, 1, 1] - e_ro[:, :, :, :]

        # Assign zero vertical gradient to directions with small elevation gradient, elsewhere calculate gradient
        grad = xarray.where(
                abs(e_cdiff.expand_dims(dim={"time": arr.shape[0]}, axis=0)) <= min_elev_diff,
                0, a_cdiff / (e_cdiff / 100))
        grad = xarray.where(a_cdiff.isnull() == True, numpy.nan, grad)

        # Reassign NaNs at the centre of the rolling windows (to avoid bias)
        grad[:, :, :, 1, 1] = numpy.nan

        # Calculate the mean vertical gradient of the moving window (where we have enough valid values)
        grad_out = xarray.where(grad.count(dim=["ry", "rx"]) >= min_ne_no,
                                grad.mean(dim=["ry", "rx"]), numpy.nan)

        # Remove invalid positive vertical gradients
        # This is incorrect for albedo, so the DEM needs to be inverted at the input stage to this function
        grad_out = xarray.where(
            grad_out > 0, numpy.nan, grad_out)

        # Interpolate data gaps (using bilinear method to get a smooth field); encode crs and nodata
        grad_out.rio.write_nodata(-999999, encoded=True, inplace=True)
        grad_out.rio.write_crs(OutCRS, inplace=True)

        grad_out = grad_out.rio.interpolate_na(method="linear")

        # Make sure there are no numerical artifacts (i.e. positive values) after interpolation
        grad_out = xarray.where(
            grad_out > 0, numpy.nan, grad_out)

        # Fill in the remaining gaps (i.e. artifacts and areas outside convex hull); encode crs and nodata
        grad_out.rio.write_nodata(-999999, encoded=True, inplace=True)
        grad_out.rio.write_crs(OutCRS, inplace=True)

        grad_out = grad_out.rio.interpolate_na(method="nearest")

        return grad_out


    # Calculate localised vertical gradients of runoff and albedo
    ##############################################################
    RU_ice_grad = vertgrad8n(RU_ice, Height, MinVertDiff, MinNeighNo)
    RU_land_grad = vertgrad8n(RU_land, Height, MinVertDiff, MinNeighNo)

    # Invert elevations for the vertical albedo gradient calculations;
    # so only meaningful vertical gradients are retained by the vertgrad8n function
    AL_ice_grad = vertgrad8n(AL_ice, Height * -1, MinVertDiff, MinNeighNo)

    # Invert the albedo gradients so that the results are physically meaningful
    AL_ice_grad = AL_ice_grad * -1


    # Save output
    ############
    # Construct container
    MAR_out = xarray.Dataset(
        data_vars={
            "Mask": (["y", "x"], Mask.data),
            "Height": (["y", "x"], Height.data),
            "RU_ice": (["time", "y", "x"], RU_ice.data),
            "RU_land": (["time", "y", "x"], RU_land.data),
            "AL_ice": (["time", "y", "x"], AL_ice.data),
            "RU_ice_grad": (["time", "y", "x"], RU_ice_grad.data),
            "RU_land_grad": (["time", "y", "x"], RU_land_grad.data),
            "AL_ice_grad": (["time", "y", "x"], AL_ice_grad.data),
        },
        coords={
            "time": ("time", RU_ice["time"].data),
            "x": ("x", RU_ice["x"].data),
            "y": ("y", RU_ice["y"].data)
        }
    )

    # Assign the dataarray CRS to the new dataset
    MAR_out = MAR_out.rio.write_crs(OutCRS, inplace=True)

    # Construct output filename
    FilePath_out = os.path.join(paths["MAROutDIR_MARDaily"], paths["CurrentDomainSTR"] + "_MAR_Input_" + CurrentYear + ".nc")

    # Save the dataset with compression levels the same as the input
    MAR_out.to_netcdf(path=FilePath_out, mode="w", format="NETCDF4",
                      encoding={
                          "Mask": {"zlib": True, "complevel": 2},
                          "Height": {"zlib": True, "complevel": 2},
                          "RU_ice": {"zlib": True, "complevel": 2},
                          "RU_land": {"zlib": True, "complevel": 2},
                          "AL_ice": {"zlib": True, "complevel": 2},
                          "RU_ice_grad": {"zlib": True, "complevel": 2},
                          "RU_land_grad": {"zlib": True, "complevel": 2},
                          "AL_ice_grad": {"zlib": True, "complevel": 2},
                      })


    # Get summary stats and rasters for the current year
    ######################################################
    # Get annual runoff rasters for ice and land, then save to geotiff
    RU_ice_annual = MAR_out["RU_ice"].sum(dim="time", skipna=False)
    RU_land_annual = MAR_out["RU_land"].sum(dim="time", skipna=False)

    RU_ice_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_IceRunoff"], paths["CurrentDomainSTR"] + "_MAR_IceRunoff_" + CurrentYear + ".tif"))
    RU_land_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_LandRunoff"], paths["CurrentDomainSTR"] + "_MAR_LandRunoff_" + CurrentYear + ".tif"))

    # Get annual mean albedo, then save to geotiff
    AL_ice_annual = MAR_out["AL_ice"].mean(dim="time", skipna=False)

    AL_ice_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_IceAlbedo"], paths["CurrentDomainSTR"] + "_MAR_IceAlbedo_" + CurrentYear + ".tif"))

    # Get annual mean vertical gradients, then save to geotiff
    RU_ice_grad_annual = MAR_out["RU_ice_grad"].mean(dim="time", skipna=False)
    RU_land_grad_annual = MAR_out["RU_land_grad"].mean(dim="time", skipna=False)
    AL_ice_grad_annual = MAR_out["AL_ice_grad"].mean(dim="time", skipna=False)

    RU_ice_grad_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_IceRunoffGrad"], paths["CurrentDomainSTR"] + "_MAR_IceRunoffGrad_" + CurrentYear + ".tif"))
    RU_land_grad_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_LandRunoffGrad"], paths["CurrentDomainSTR"] + "_MAR_LandRunoffGrad_" + CurrentYear + ".tif"))
    AL_ice_grad_annual.rio.to_raster(os.path.join
        (paths["MAROutDIR_IceAlbedoGrad"], paths["CurrentDomainSTR"] + "_MAR_IceAlbedoGrad_" + CurrentYear + ".tif"))

    print("Year: " + CurrentYear + " processed", flush=True)


# Initiate parallel processing
#################################
# Get input file list
DomainDictInit = {"RGI_3_CanadaN": "Canadian_Arctic",
              "RGI_4_CanadaS": "Canadian_Arctic",
              "RGI_5_Greenland": "Greenland",
              "RGI_6_Iceland": "Iceland",
              "RGI_7_Svalbard": "Russian_Arctic",
              "RGI_9_RussiaN": "Russian_Arctic"}

SearchArg = os.path.join(InputDIR, "*" + DomainDictInit[CurrentDomainSTR] + "*" + ".nc")
filelist_in = glob.glob(SearchArg)
filelist_in.sort(reverse=False)

# Start parallel pool
if __name__ == "__main__":
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool(processes=core_no)

    print(pool, flush=True)

    for count_y, file_in in enumerate(filelist_in):
        pool.apply_async(process_annual_mar_data, args=(count_y, file_in, dir_dict))

    pool.close()
    pool.join()

print("Finished", flush=True)
