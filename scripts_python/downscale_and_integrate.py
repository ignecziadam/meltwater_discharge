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
import numpy
import math
import geopandas
import pandas
import multiprocessing
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator


# Initialise script
####################
# Get input and output directory paths from the bash shell
working_dir = os.getenv("WORKING_DIR")  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"

# Get positional arguments from the bash shell
current_domain_str = sys.argv[1]  # sys.argv[1] "RGI_5_Greenland"

# Set the number of cores to use in parallel processing
core_no = sys.argv[2]  # sys.argv[2] 1
core_no = int(core_no)


# Check if output directories exists, if not create them
#########################################################
# Root results directory
results_dir = os.path.join(working_dir, current_domain_str, "Results")

# Subdirectories
results_dir_IR = os.path.join(results_dir, "IceRunoff")
results_dir_IR_BSL = os.path.join(results_dir, "IceRunoffBSL")
results_dir_LR = os.path.join(results_dir, "LandRunoff")
results_dir_IA = os.path.join(results_dir, "IceAlbedo")
results_dir_basins = os.path.join(results_dir, "Basins")
results_dir_totalRU = os.path.join(results_dir, "TotalRunoff")

if not os.path.lexists(results_dir):
    os.mkdir(results_dir)

if not os.path.lexists(results_dir_IR):
    os.mkdir(results_dir_IR)
if not os.path.lexists(results_dir_IR_BSL):
    os.mkdir(results_dir_IR_BSL)
if not os.path.lexists(results_dir_LR):
    os.mkdir(results_dir_LR)
if not os.path.lexists(results_dir_IA):
    os.mkdir(results_dir_IA)
if not os.path.lexists(results_dir_basins):
    os.mkdir(results_dir_basins)
if not os.path.lexists(results_dir_totalRU):
    os.mkdir(results_dir_totalRU)

# Set up directory dictionary
dir_dict = {"working_dir": working_dir,
         "current_domain_str": current_domain_str,
         "results_dir": results_dir,
         "results_dir_IR": results_dir_IR,
         "results_dir_IR_BSL": results_dir_IR_BSL,
         "results_dir_LR": results_dir_LR,
         "results_dir_IA": results_dir_IA,
         "results_dir_basins": results_dir_basins,
         "results_dir_totalRU": results_dir_totalRU}


# Define a function that processes a year of data
##################################################
def process_annual_data(itno, file, paths):
    # Display current year
    current_year = file.split("_")[-1].split(".")[0]
    print("Year: " + current_year + " start processing", flush=True)


    # Load static the datasets
    ############################
    dem = rioxarray.open_rasterio(
        os.path.join(paths["working_dir"], paths["current_domain_str"],
                     paths["current_domain_str"] + "_DEM.tif"),
        band_as_variable=False)
    dem = dem.squeeze(dim="band")

    # Load basins
    basins = rioxarray.open_rasterio(
        os.path.join(paths["working_dir"], paths["current_domain_str"],
                     paths["current_domain_str"] + "_Basins.tif")
    )
    basins = basins.squeeze(dim="band")

    # Load land mask
    land_mask = rioxarray.open_rasterio(
        os.path.join(paths["working_dir"], paths["current_domain_str"],
                     paths["current_domain_str"] + "_LandMask.tif")
    )
    land_mask = land_mask.squeeze(dim="band")

    # Load ice mask
    ice_mask = rioxarray.open_rasterio(
        os.path.join(paths["working_dir"], paths["current_domain_str"],
                     paths["current_domain_str"] + "_IceMask.tif")
    )
    ice_mask = ice_mask.squeeze(dim="band")

    # Load basins domain
    basins_domain = geopandas.read_file(
        os.path.join(paths["working_dir"], paths["current_domain_str"], "selectors",
                     paths["current_domain_str"] + "_Basins_extent_laeaa.shp"
                     )
    )

    # Clip the datasets with the basins domain
    dem = dem.rio.clip(basins_domain.geometry.values, all_touched=False)
    basins = basins.rio.clip(basins_domain.geometry.values, all_touched=False)
    land_mask = land_mask.rio.clip(basins_domain.geometry.values, all_touched=False)
    ice_mask = ice_mask.rio.clip(basins_domain.geometry.values, all_touched=False)

    dem = dem.where(dem != -9999)
    dem.rio.write_nodata(-9999, encoded=True, inplace=True)

    basins = basins.where(basins != -1)
    basins.rio.write_nodata(-1, encoded=True, inplace=True)

    land_mask = land_mask.where(land_mask != -1)
    land_mask.rio.write_nodata(-1, encoded=True, inplace=True)

    ice_mask = ice_mask.where(ice_mask != -1)
    ice_mask.rio.write_nodata(-1, encoded=True, inplace=True)

    # Load the annual dataset
    ##########################
    mar_in = xarray.open_dataset(file, decode_coords="all").load()

    # Define CRS for rioxarray
    mar_in.rio.write_crs(mar_in.spatial_ref.crs_wkt, inplace=True)


    # Set up containers for the results
    ####################################
    # Set up data arrays for 2D outputs
    annual_2D_ice_runoff = xarray.zeros_like(dem)
    annual_2D_ice_runoff_BSL = xarray.zeros_like(dem)
    annual_2D_land_runoff = xarray.zeros_like(dem)

    annual_2D_avg_albedo = xarray.zeros_like(dem)

    # Set up lists for 1D outputs (ice runoff, ice runoff below snowline, land runoff)
    daily_1D_icerunoff_raw = []
    daily_1D_icerunoff_upsample = []
    daily_1D_icerunoff_downscale = []

    daily_1D_icerunoffBSL_raw = []
    daily_1D_icerunoffBSL_upsample = []
    daily_1D_icerunoffBSL_downscale = []

    daily_1D_landrunoff_raw = []
    daily_1D_landrunoff_upsample = []
    daily_1D_landrunoff_downscale = []


    # Set up dataframe for basin specific 1D outputs
    basin_ID = numpy.unique(basins)
    basin_ID = basin_ID[~numpy.isnan(basin_ID)]

    date = pandas.to_datetime(mar_in["time"]).date

    daily_1D_icerunoff_basin = numpy.zeros((date.size, basin_ID.size))
    daily_1D_icerunoffBSL_basin = numpy.zeros((date.size, basin_ID.size))
    daily_1D_landrunoff_basin = numpy.zeros((date.size, basin_ID.size))


    # Create xy grids
    ##################
    # Native MAR x,y grid
    x_mar_grid, y_mar_grid = numpy.meshgrid(mar_in["x"].values, mar_in["y"].values, indexing="xy")

    # Low resolution grid for nearest neighbour extrapolation, snap this to the original x,y grids
    x_start_no = round((dem["x"].values[0] - mar_in["x"].values[0]) / mar_in.rio.resolution()[0], 0)
    x_end_no = round((dem["x"].values[-1] - mar_in["x"].values[-1]) / mar_in.rio.resolution()[0], 0)
    y_start_no = round((dem["y"].values[0] - mar_in["y"].values[0]) / mar_in.rio.resolution()[1], 0)
    y_end_no = round((dem["y"].values[-1] - mar_in["y"].values[-1]) / mar_in.rio.resolution()[1], 0)

    x_start = mar_in["x"].values[0] + x_start_no * mar_in.rio.resolution()[0]
    x_end = mar_in["x"].values[-1] + x_end_no * mar_in.rio.resolution()[0]
    y_start = mar_in["y"].values[0] + y_start_no * mar_in.rio.resolution()[1]
    y_end = mar_in["y"].values[-1] + y_end_no * mar_in.rio.resolution()[1]

    x_lowres = numpy.arange(x_start - mar_in.rio.resolution()[0],
                            x_end + 2 * mar_in.rio.resolution()[0],
                            mar_in.rio.resolution()[0])
    y_lowres = numpy.arange(y_start - mar_in.rio.resolution()[1],
                            y_end + 2 * mar_in.rio.resolution()[1],
                            mar_in.rio.resolution()[1])

    x_lowres_grid, y_lowres_grid = numpy.meshgrid(x_lowres, y_lowres, indexing="xy")

    # High resolution grid for linear interpolation
    x_highres = numpy.arange(dem["x"].values[0], dem["x"].values[-1] + dem.rio.resolution()[0],
                             dem.rio.resolution()[0])
    y_highres = numpy.arange(dem["y"].values[0], dem["y"].values[-1] + dem.rio.resolution()[1],
                             dem.rio.resolution()[1])

    x_highres_grid, y_highres_grid = numpy.meshgrid(x_highres, y_highres, indexing="xy")


    # Set up function for upsampling (inputs ndarrays)
    #######################################################
    def upsample(var_input, x_input, y_input, x_fill, y_fill, x_high, y_high):
        # Only retain valid data for the nearest neighbour interpolator
        x = x_input.flatten()[
            ~numpy.isnan(var_input.flatten())
        ]
        y = y_input.flatten()[
            ~numpy.isnan(var_input.flatten())
        ]
        var = var_input.flatten()[
            ~numpy.isnan(var_input.flatten())
        ]

        # Nearest neighbour interpolation; this handles filling data gaps and extrapolating outside the MAR domain
        interp_NN = NearestNDInterpolator((y, x), var)
        var_filled = interp_NN((y_fill, x_fill))

        # Bilinear interpolation; this provides high resolution, smooth downscaled data
        interp_LIN = RegularGridInterpolator((y_fill[:, 0], x_fill[0, :]), var_filled,
                                             method="linear", bounds_error=False, fill_value=None)
        var_highres = interp_LIN((y_high, x_high))

        return var_highres


    # Upsample DEM
    ################
    model_dem_highres = upsample(mar_in["Height"].values,
                                 x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

    # Calculate the difference between model dem and high-res dem
    dem = xarray.where(land_mask == 1, dem, numpy.nan)
    dem_diff = dem - model_dem_highres


    # Iterate through the days of the current year
    ##############################################
    for count_d, day in enumerate(mar_in["time"]):

        # Print a message after 10 processed days
        if (count_d / 10 - math.floor(count_d / 10)) == 0:
            print("Year " + current_year + ": " + str(count_d) + " days processed", flush=True)


        # Downscale ice albedo
        #######################
        # Upsample albedo and its vertical gradient
        albedo_highres = upsample(mar_in["AL_ice"].sel(time=day).values,
                                  x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        albedo_grad_highres = upsample(mar_in["AL_ice_grad"].sel(time=day).values,
                                       x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        # Apply vertical gradients
        albedo = albedo_highres + ((dem_diff / 100) * albedo_grad_highres)

        # Mask out non-ice areas and set albedo outside [0, 1] to valid min and max values
        albedo = xarray.where(ice_mask == 1, albedo, numpy.nan)
        albedo = xarray.where(albedo < 0, 0, albedo)
        albedo = xarray.where(albedo > 1, 1, albedo)


        # Downscale ice runoff
        #######################
        # Upsample ice runoff and its vertical gradient
        ice_runoff_highres = upsample(mar_in["RU_ice"].sel(time=day).values,
                                      x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        ice_runoff_grad_highres = upsample(mar_in["RU_ice_grad"].sel(time=day).values,
                                           x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        # Apply vertical gradients
        ice_runoff = ice_runoff_highres + ((dem_diff / 100) * ice_runoff_grad_highres)

        # Mask out non-ice areas and set negative runoff to zero
        ice_runoff = xarray.where(ice_mask == 1, ice_runoff, numpy.nan)
        ice_runoff = xarray.where(ice_runoff < 0, 0, ice_runoff)

        # Get runoff for the zone below the snowline
        ice_runoff_BSL = xarray.where(albedo <= 0.7, ice_runoff, 0)


        # Downscale land runoff
        #######################
        # Upsample ice runoff and its vertical gradient
        land_runoff_highres = upsample(mar_in["RU_land"].sel(time=day).values,
                                       x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        land_runoff_grad_highres = upsample(mar_in["RU_land_grad"].sel(time=day).values,
                                            x_mar_grid, y_mar_grid, x_lowres_grid, y_lowres_grid, x_highres_grid, y_highres_grid)

        # Apply vertical gradients
        land_runoff = land_runoff_highres + ((dem_diff / 100) * land_runoff_grad_highres)

        # Mask out ice areas and set negative runoff to zero
        land_runoff = xarray.where(ice_mask == 1, numpy.nan, land_runoff)
        land_runoff = xarray.where(land_runoff < 0, 0, land_runoff)


        # Calculate summary statistics
        ###############################
        # Add the current day to the cumulative 2D arrays
        annual_2D_ice_runoff = annual_2D_ice_runoff + ice_runoff
        annual_2D_ice_runoff_BSL = annual_2D_ice_runoff_BSL + ice_runoff_BSL
        annual_2D_land_runoff = annual_2D_land_runoff + land_runoff

        annual_2D_avg_albedo = annual_2D_avg_albedo + albedo

        # Calculate pixel area
        highres_pix_area = abs(ice_runoff.rio.resolution()[0] * ice_runoff.rio.resolution()[1])
        mar_pix_area = abs(mar_in.rio.resolution()[0] * mar_in.rio.resolution()[1])

        # Calculate raw total daily runoff products
        tmp = (mar_in["RU_ice"].sel(time=day) *
               (mar_in["Mask"] / 100)
               )
        tmp = tmp.rio.clip(basins_domain.geometry.values, all_touched=False)
        daily_1D_icerunoff_raw.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * mar_pix_area
            )
        )

        tmp = (mar_in["RU_ice"].sel(time=day) *
               (mar_in["Mask"] / 100) *
               xarray.where(mar_in["AL_ice"].sel(time=day) <= 0.7, 1, 0)
               )
        tmp = tmp.rio.clip(basins_domain.geometry.values, all_touched=False)
        daily_1D_icerunoffBSL_raw.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * mar_pix_area
            )
        )

        tmp = (mar_in["RU_land"].sel(time=day) *
               ((100 - mar_in["Mask"]) / 100) *
               (xarray.where(mar_in["Height"] == 0, 0, 1) - mar_in["Height"].isnull())
               )
        tmp = tmp.rio.clip(basins_domain.geometry.values, all_touched=False)
        daily_1D_landrunoff_raw.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * mar_pix_area
            )
        )

        # Calculate upsampled total daily runoff products
        tmp = (ice_runoff_highres *
               xarray.where(ice_mask == 1, 1, numpy.nan))
        daily_1D_icerunoff_upsample.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )

        tmp = (ice_runoff_highres *
               xarray.where(ice_mask == 1, 1, numpy.nan) *
               xarray.where(albedo_highres <= 0.7, 1, numpy.nan))
        daily_1D_icerunoffBSL_upsample.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )

        tmp = (land_runoff_highres *
               xarray.where(ice_mask == 1, numpy.nan, 1) *
               xarray.where(land_mask == 1, 1, numpy.nan))
        daily_1D_landrunoff_upsample.append(
            float(
                tmp.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )

        # Calculate downscaled total daily runoff products
        daily_1D_icerunoff_downscale.append(
            float(
                ice_runoff.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )

        daily_1D_icerunoffBSL_downscale.append(
            float(
                ice_runoff_BSL.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )

        daily_1D_landrunoff_downscale.append(
            float(
                land_runoff.sum(dim=["x", "y"]) / 1000 * highres_pix_area
            )
        )


        # Calculate basin specific total runoff
        ########################################
        daily_1D_icerunoff_basin[count_d, :] = \
            ice_runoff.groupby(basins).sum().data / 1000 * highres_pix_area
        daily_1D_icerunoffBSL_basin[count_d, :] = \
            ice_runoff_BSL.groupby(basins).sum().data / 1000 * highres_pix_area
        daily_1D_landrunoff_basin[count_d, :] = \
            land_runoff.groupby(basins).sum().data / 1000 * highres_pix_area


    # Summarise for the current year and save
    ##########################################
    # Save annual total runoff and average albedo rasters
    annual_2D_ice_runoff.rio.to_raster(
        os.path.join(paths["results_dir_IR"],
                     paths["current_domain_str"] + "_IceRunoffDownscaled_" + current_year + ".tif"),
        compress='LZW'
    )
    annual_2D_ice_runoff_BSL.rio.to_raster(
        os.path.join(paths["results_dir_IR_BSL"],
                     paths["current_domain_str"] + "_IceRunoffBSLDownscaled_" + current_year + ".tif"),
        compress='LZW'
    )
    annual_2D_land_runoff.rio.to_raster(
        os.path.join(paths["results_dir_LR"],
                     paths["current_domain_str"] + "_LandRunoffDownscaled_" + current_year + ".tif"),
        compress='LZW'
    )

    annual_2D_avg_albedo = annual_2D_avg_albedo / date.size
    annual_2D_avg_albedo.rio.to_raster(
        os.path.join(paths["results_dir_IA"],
                     paths["current_domain_str"] + "_AvgAlbedoDownscaled_" + current_year + ".tif"),
        compress='LZW'
    )

    # Save annual total runoff
    ExportTable = pandas.DataFrame({
        "Date": list(date),
        "Raw ice runoff (m^3)": daily_1D_icerunoff_raw,
        "Raw ice runoff below snowline (m^3)": daily_1D_icerunoffBSL_raw,
        "Raw land runoff (m^3)": daily_1D_landrunoff_raw,
        "Upsampled ice runoff (m^3)": daily_1D_icerunoff_upsample,
        "Upsampled ice runoff below snowline (m^3)": daily_1D_icerunoffBSL_upsample,
        "Upsampled land runoff (m^3)": daily_1D_landrunoff_upsample,
        "Downscaled ice runoff (m^3)": daily_1D_icerunoff_downscale,
        "Downscaled ice runoff below snowline (m^3)": daily_1D_icerunoffBSL_downscale,
        "Downscaled land runoff (m^3)": daily_1D_landrunoff_downscale
    })

    ExportTable.to_csv(
        os.path.join(paths["results_dir_totalRU"],
                     paths["current_domain_str"] + "_TotalRunoff_" + current_year + ".txt"),
        index=False
    )

    # Save basin specific results
    BasinRunoffOut = xarray.Dataset(
        data_vars={
            "IceRunoff": (["time", "BasinID"], daily_1D_icerunoff_basin),
            "IceRunoff_BSL": (["time", "BasinID"], daily_1D_icerunoffBSL_basin),
            "LandRunoff": (["time", "BasinID"], daily_1D_landrunoff_basin)
        },
        coords={
            "time": ("time", mar_in["time"].data),
            "BasinID": ("BasinID", basin_ID)
        }
    )

    FilePath_out = os.path.join(paths["results_dir_basins"],
                                paths["current_domain_str"] + "_BasinRunoff" + current_year + ".nc")
    BasinRunoffOut.to_netcdf(path=FilePath_out, mode="w", format="NETCDF4",
                             encoding={
                                 "IceRunoff": {"zlib": True, "complevel": 2},
                                 "IceRunoff_BSL": {"zlib": True, "complevel": 2},
                                 "LandRunoff": {"zlib": True, "complevel": 2},
                             })

    print("Year: " + current_year + " finished processing", flush=True)


# Initiate parallel processing of the defined function for the annual data
############################################################################
# Set up the input file list
mar_search_arg = os.path.join(working_dir, current_domain_str, "MAR_Preprocessing", "MAR_Daily",
                              current_domain_str + "_MAR_Input_" + "*" + ".nc")
filelist_in = glob.glob(mar_search_arg)
filelist_in.sort(reverse=False)

# Initiate parallel processing
if __name__ == "__main__":
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool(processes=core_no)

    print(pool, flush=True)

    for count_y, file_in in enumerate(filelist_in):
        pool.apply_async(process_annual_data, args=(count_y, file_in, dir_dict))

    pool.close()
    pool.join()

    print("Finished processing for " + current_domain_str)
