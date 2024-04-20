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
import pandas
import geopandas


# Initialise script
####################
# Get input and output directory paths from the bash shell
working_dir = os.getenv("WORKING_DIR")  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"

# Get positional arguments from the bash shell
current_domain_str = sys.argv[1]  # sys.argv[1] "RGI_6_Iceland"

# Root results directory
results_dir = os.path.join(working_dir, current_domain_str, "Results")
results_dir_basins = os.path.join(results_dir, "Basins")
results_dir_TRU = os.path.join(results_dir, "TotalRunoff")

print("Start processing " + current_domain_str, flush=True)

# Create input file lists
##########################
# Set up the input file list for total runoff
TRU_filelist = glob.glob(
    os.path.join(results_dir_TRU,
                 current_domain_str + "_TotalRunoff_" + "*" + ".txt")
)

TRU_filelist.sort(reverse=False)

# Set up the input file list for basins
basins_filelist = glob.glob(
    os.path.join(results_dir_basins,
                 current_domain_str + "_BasinRunoff" + "*" + ".nc")
)

basins_filelist.sort(reverse=False)

# Load in the outflow points
outflow_points = geopandas.read_file(
    os.path.join(
        working_dir, current_domain_str, current_domain_str + "_OutflowPoints.shp")
)


# Concatenate annual total runoff
##################################
# Set up a container (a list) to hold the sequence (1/yr) of total runoff dataframes
TRU_seq = []

# Load in the files to the container
for file in TRU_filelist:
    TRU_seq.append(
        pandas.read_csv(file)
    )

# Use the built-in concatenate function from pandas
TRU_complete = pandas.concat(TRU_seq, axis=0)

print("Total runoff concatenated", flush=True)

# Concatenate basin runoff
###########################
# Set up a container (a list) to hold the sequence (1/yr) of total runoff dataframes
basins_seq = []

# Load in the files to the container
for file in basins_filelist:
    basins_seq.append(
        xarray.open_dataset(file)
    )

# Use the built-in concatenate function from pandas
basins_complete = xarray.concat(basins_seq, dim="time")

print("Basin specific runoff concatenated", flush=True)

# Add the geographical coordinates of outflow points
basins_complete = basins_complete.assign_coords(
    {"lat": ("BasinID", outflow_points["LAT"]), "lon": ("BasinID", outflow_points["LON"]),
     "x": ("BasinID", outflow_points.geometry.x), "y": ("BasinID", outflow_points.geometry.y)}
)


# Save the final results
#########################
# Total runoff
TRU_complete.to_csv(
    os.path.join(results_dir, current_domain_str + "_TotalRunoff.txt"),
    index=False
)

# Basin specific runoff
basins_complete_filename = os.path.join(results_dir,
                                        current_domain_str + "_BasinRunoff.nc")

basins_complete.to_netcdf(
    path=basins_complete_filename, mode="w", format="NETCDF4",
    encoding={
        "IceRunoff": {"zlib": True, "complevel": 2},
        "IceRunoff_BSL": {"zlib": True, "complevel": 2},
        "LandRunoff": {"zlib": True, "complevel": 2},
    }
)

print("Finished processing " + current_domain_str, flush=True)
