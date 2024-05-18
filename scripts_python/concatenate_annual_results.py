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
import numpy
import pandas
import geopandas
from datetime import datetime


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
        working_dir, current_domain_str, current_domain_str + "_OutflowPoints.gpkg")
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


# Save the final results
#########################
# Total runoff
TRU_complete.to_csv(
    os.path.join(results_dir, current_domain_str + "_TotalRunoff.txt"),
    index=False
)

# Ensure that basin IDs are integers
basins_complete["BasinID"] = basins_complete["BasinID"].astype("int32")

# Prepare time coordinate
time_units = 'days since {Y}-{M}-{D} 00:00:00'.format(
    Y=int(basins_complete["time"].dt.year[0]),
    M=int(basins_complete["time"].dt.month[0]),
    D=int(basins_complete["time"].dt.day[0]),

)
time_calendar = basins_complete["time"].dt.calendar
times = pandas.date_range(
    start=str(int(basins_complete["time"].dt.year[0])) +
          "-" + str(int(basins_complete["time"].dt.month[0])) +
          "-" + str(int((basins_complete["time"].dt.day[0]))),
    end=str(int(basins_complete["time"].dt.year[-1])) +
          "-" + str(int(basins_complete["time"].dt.month[-1])) +
          "-" + str(int(basins_complete["time"].dt.day[-1])),
    freq='1D',
)

# Construct compliant Xarray Dataset
basins_export = xarray.Dataset(
    data_vars={
        "IceRunoff": (["time", "BasinID"], basins_complete["IceRunoff"].data,
            {"long_name": "Meltwater discharge from ice runoff",
             "standard_name": "water_volume_transport_into_sea_water_from_rivers",
             "units": "m3",
             "coverage_content_type": "modelResult"
              }),
        "IceRunoff_BSL": (["time", "BasinID"], basins_complete["IceRunoff_BSL"].data,
            {"long_name": "Meltwater discharge from ice runoff below the snowline",
             "standard_name": "water_volume_transport_into_sea_water_from_rivers",
             "units": "m3",
             "coverage_content_type": "modelResult"
             }),
        "LandRunoff": (["time", "BasinID"], basins_complete["LandRunoff"].data,
            {"long_name": "Meltwater discharge from land runoff",
             "standard_name": "water_volume_transport_into_sea_water_from_rivers",
             "units": "m3",
             "coverage_content_type": "modelResult"
             })
    },
    coords={
        "time": ("time", times,
            {"long_name": "time",
             "standard_name": "time"
             }),
        "BasinID": ("BasinID", basins_complete["BasinID"].data,
            {"long_name": "Drainage basin ID"
             }),
        "x": ("BasinID", outflow_points.geometry.x,
            {"long_name": "Easting",
             "standard_name": "projection_x_coordinate",
             "units": "m"
             }),
        "y": ("BasinID", outflow_points.geometry.y,
            {"long_name": "Northing",
             "standard_name": "projection_y_coordinate",
             "units": "m"
             }),
        "lat": ("BasinID", outflow_points["LAT"],
            {"long_name": "latitude",
             "standard_name": "latitude",
             "units": "degrees_north"
             }),
        "lon": ("BasinID", outflow_points["LON"],
            {"long_name": "longitude",
             "standard_name": "longitude",
             "units": "degrees_east"
             }),
    },
    attrs={
        "title": "Pan-Arctic land-ice and tundra meltwater discharge database from 1950 to 2021",
        "summary": "A high resolution (daily, 250m) land ice and tundra meltwater discharge dataset for the period "
                   "1950-2021. Meltwater discharge is derived from daily ~6 km regional climate model, "
                   "Modéle Atmosphérique Régional (MAR), runoff simulations that are statistically downscaled and "
                   "routed to the coastlines.The statistical downscaling algorithm uses native vertical gradients of "
                   "the MAR data and high resolution (250 m) DEM, land mask (Copernicus GLO-90) and ice mask (GIMP, "
                   "RGI) datasets.Routing to coastal outflow points is performed by a hydrological routing scheme "
                   "applied to the high-resolution DEM and the downscaled runoff. Meltwater components from "
                   "non-glaciated land, bare glacier ice and glaciated area above the snowline are separated to "
                   "facilitate further analysis.",
        "keywords": "Arctic, Discharge, Meltwater, North Atlantic Ocean, Canada, Greenland, Iceland, Russian Arctic, "
                    "Svalbard",
        "institution": "University of Bristol",
        "creator_name": "Adam Igneczi, Jonathan Bamber",
        "Conventions": "CF-1.8, ACDD-1.3",
        "date_created": str(datetime.now()),
        "history": str(datetime.now()),
        "project": "Pan-Arctic observing System of Systems: Implementing Observations for societal Needs (Arctic "
                   "PASSION)",
        "acknowledgement": "Horizon 2020 (H2020), grant no. 101003472; German Federal Ministry of Education and "
                           "Research (BMBF), grant no. 01DD20001",
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "references": "https://github.com/ignecziadam/meltwater_discharge.git",
        "projected_crs_name": "EPSG:3574",
        "geographical_crs_name": "EPSG:4326"
    }
)

basins_export.time.encoding["units"] = time_units

# Construct filename
basins_export_filename = os.path.join(results_dir,
                                        current_domain_str + "_BasinRunoff.nc")

# Write to file
basins_export.to_netcdf(
    path=basins_export_filename, mode="w", format="NETCDF4",
    encoding={
        "IceRunoff": {"zlib": True, "complevel": 2},
        "IceRunoff_BSL": {"zlib": True, "complevel": 2},
        "LandRunoff": {"zlib": True, "complevel": 2},
    }
)

print("Finished processing " + current_domain_str, flush=True)
