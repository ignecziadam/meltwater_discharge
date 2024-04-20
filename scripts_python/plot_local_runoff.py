# -*- coding: utf-8 -*-

"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

# Import modules
import os
import pandas
import xarray
import numpy
import matplotlib as mp
import matplotlib.pyplot as plt

# Set up namespace
working_dir = "C:\postdoc_bristol\gis\Regions"  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"
output_dir = os.path.join(working_dir, "BasinPlots")

# What to plot
domain = "RGI_5_Greenland"
basin = 41042

StartYear = 2019
EndYear = 2020

# Create directories and paths
if not os.path.lexists(output_dir):
    os.mkdir(output_dir)

path = os.path.join(
    working_dir, domain, "Results", domain + "_BasinRunoff.nc"
)

# Import data
domain_data = xarray.open_dataset(path)

# Select data
RU_ice = domain_data["IceRunoff"].sel(BasinID=basin)
RU_iceBSL = domain_data["IceRunoff_BSL"].sel(BasinID=basin)
RU_land = domain_data["LandRunoff"].sel(BasinID=basin)

RU_ice = RU_ice.to_series()
RU_iceBSL = RU_iceBSL.to_series()
RU_land = RU_land.to_series()

period_sec = RU_ice.index.to_series().diff().dt.total_seconds().values
period_sec[0] = period_sec[1]

runoff = pandas.DataFrame(
    {
        "ice": RU_ice.values / period_sec,
        "ice BSL": RU_iceBSL.values / period_sec,
        "land": RU_land.values / period_sec,
    },
    index=RU_ice.index
)


runoff_movmean = runoff.rolling(window=7, min_periods=1, center=True).mean()
cm = 1/2.54

#
ax = runoff_movmean.plot(ylim=(0, 1400), xlim=(pandas.Timestamp('1950-01-01'), pandas.Timestamp('2022-01-01')),
                         figsize=(10*cm, 8*cm),
                         fontsize=8, lw=1)
ax.legend(fontsize=8)
ax.set_title("BasinID: " + str(basin), fontdict={'fontsize': 10})
ax.set_xlabel('Date', fontdict={'fontsize': 10})
ax.set_ylabel('Runoff (m^3/s)', fontdict={'fontsize': 10})

fig = plt.gcf()
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, domain + "_BasinID_" + str(basin) + "_7dayMovMeanRU.png"
    ), dpi=600)
plt.close(fig=fig)

# Subset
runoff_movmean_subset = runoff_movmean.loc[str(StartYear) + "-01-01":str(EndYear) + "-01-01"]

ax = runoff_movmean_subset.plot(ylim=(0, 1400), xlim=(pandas.Timestamp(str(StartYear) + "-01-01"), pandas.Timestamp(str(EndYear) + "-01-01")),
                                figsize=(10*cm, 8*cm),
                                fontsize=8, lw=1)

ax.legend(fontsize=8)
ax.set_title("BasinID: " + str(basin), fontdict={'fontsize': 10})
ax.set_xlabel('Date', fontdict={'fontsize': 10})
ax.set_ylabel('Runoff (m^3/s)', fontdict={'fontsize': 10})

fig = plt.gcf()
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, domain + "_BasinID_" + str(basin) + "_Subset_7dayMovMeanRU.png"
    ), dpi=600)
plt.close(fig=fig)
