# -*- coding: utf-8 -*-

"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

# Import modules
################
import os
import sys
import xarray


# Initialise script
####################
regions = ["RGI_3_CanadaN", "RGI_4_CanadaS", "RGI_5_Greenland", "RGI_6_Iceland", "RGI_7_Svalbard", "RGI_9_RussiaN"]
# "RGI_3_CanadaN", "RGI_4_CanadaS", "RGI_5_Greenland", "RGI_6_Iceland", "RGI_7_Svalbard", "RGI_9_RussiaN"

# Get input and output directory paths from the bash shell
working_dir = os.getenv("WORKING_DIR")  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"


# Load and transform data
##########################
basins_filelist = []

for region in regions:
    basins_filelist.append(
        os.path.join(working_dir, region, "Results", region + "_BasinRunoff.nc")
    )

basins_seq = []

# Open the files for each region, group for each month, and concatenate
print("Start loading and transforming data", flush=True)

for i, file in enumerate(basins_filelist):
    tmp = xarray.open_dataset(file)

    # Add the name of the region as a coordinate
    print("Processing " + regions[i], flush=True)
    region_name = [int(regions[i].split("_")[1])] * tmp["BasinID"].size
    tmp = tmp.assign_coords(
        {"RGI_region": ("BasinID", region_name)}
                            )

    # Calculate monthly runoff
    tmp = tmp.resample(time="1MS").sum()

    # Append to the array
    basins_seq.append(tmp)

basins_complete = xarray.concat(basins_seq, dim="BasinID")


# Save data
############
print("Saving data", flush=True)

basins_complete_filename = os.path.join(working_dir, "panArctic_MonthlyBasinRunoff.nc")

basins_complete.to_netcdf(
    path=basins_complete_filename, mode="w", format="NETCDF4",
    encoding={
        "IceRunoff": {"zlib": True, "complevel": 2},
        "IceRunoff_BSL": {"zlib": True, "complevel": 2},
        "LandRunoff": {"zlib": True, "complevel": 2},
    }
)

print("Finished processing", flush=True)
