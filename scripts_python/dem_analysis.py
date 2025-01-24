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
import matplotlib
import matplotlib.pyplot as plt


# Get input and output directory paths
working_dir = "C:\\postdoc_bristol\\gis\\Regions"  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"
output_dir = os.path.join(working_dir, "Plots")

# set up namespace and data containers
domains = [
    "RGI_3_CanadaN",
    "RGI_4_CanadaS",
    "RGI_5_Greenland",
    "RGI_6_Iceland",
    "RGI_7_Svalbard",
    "RGI_9_RussiaN"
]

ice_stats_list = []
land_stats_list = []
ice_bins_list = []
land_bins_list = []

# load the data
for name in domains:
    current_domain_str = name

    # load basins domain
    basins_domain = geopandas.read_file(
        os.path.join(working_dir, current_domain_str, "selectors",
                     current_domain_str + "_Basins_extent_laeaa.shp"
                     )
    )

    # load dems and masks
    dem_diff = rioxarray.open_rasterio(
        os.path.join(working_dir, current_domain_str, current_domain_str + "_DemDiff.tif"),
        band_as_variable=False)
    dem_diff = dem_diff.squeeze(dim="band")

    dem_mar = rioxarray.open_rasterio(
        os.path.join(working_dir, current_domain_str, current_domain_str + "_MAR_highresDem.tif"),
        band_as_variable=False)
    dem_mar = dem_mar.squeeze(dim="band")

    ice_mask = rioxarray.open_rasterio(
        os.path.join(working_dir, current_domain_str, current_domain_str + "_IceMask.tif"),
        band_as_variable=False)
    ice_mask = ice_mask.squeeze(dim="band")
    ice_mask = ice_mask.rio.clip(basins_domain.geometry.values, all_touched=False)

    land_mask = rioxarray.open_rasterio(
        os.path.join(working_dir, current_domain_str, current_domain_str + "_LandMask.tif"),
        band_as_variable=False)
    land_mask = land_mask.squeeze(dim="band")
    land_mask = land_mask.rio.clip(basins_domain.geometry.values, all_touched=False)

    # mask the dems
    dem_diff_ice = xarray.where(ice_mask.data == 1, dem_diff.data, numpy.nan)
    dem_mar_ice = numpy.where(ice_mask.data == 1, dem_mar.data, numpy.nan)

    dem_diff_land = numpy.where(ice_mask.data == 0, dem_diff.data, numpy.nan)
    dem_mar_land = numpy.where(ice_mask.data == 0, dem_mar.data, numpy.nan)

    # determine the bin edges for the MAR elevations
    highres_pix_area = abs(dem_diff.rio.resolution()[0] * dem_diff.rio.resolution()[1])

    ice_bins = numpy.linspace(float(numpy.nanmin(dem_mar_ice)),
                              float(numpy.nanmax(dem_mar_ice)), 20)

    land_bins = numpy.linspace(float(numpy.nanmin(dem_mar_land)),
                               float(numpy.nanmax(dem_mar_land)), 20)

    # assign the data to the bins
    dem_mar_ice_cat = numpy.digitize(dem_mar_ice, ice_bins)
    dem_mar_ice_cat = numpy.where(ice_mask.data == 1, dem_mar_ice_cat, numpy.nan)

    dem_mar_ice_cat = dem_mar_ice_cat.flatten()
    dem_diff_ice = dem_diff_ice.flatten()

    dem_mar_land_cat = numpy.digitize(dem_mar_land, land_bins)
    dem_mar_land_cat = xarray.where(ice_mask.data == 0, dem_mar_land_cat, numpy.nan)

    dem_mar_land_cat = dem_mar_land_cat.flatten()
    dem_diff_land = dem_diff_land.flatten()

    # create dataframes for grouping
    ice = pandas.DataFrame({"category": dem_mar_ice_cat,
                            "dem_diff": dem_diff_ice})

    land = pandas.DataFrame({"category": dem_mar_land_cat,
                            "dem_diff": dem_diff_land})

    # group the COP-250 DEM minus MAR DEM data according to MAR elevation bins
    ice_stats = pandas.DataFrame(columns=["area", "med_dem_dif"],
                                 index=range(1, len(ice_bins), 1))
    ice_stats["area"] = (ice.groupby(['category']).count() * highres_pix_area / 1000000).dem_diff
    ice_stats["med_dem_dif"] = (ice.groupby(['category']).median()).dem_diff
    ice_stats["med_dem_dif"] = ice_stats["med_dem_dif"].where(ice_stats["area"] >= ice_stats["area"].sum() / 1000)

    land_stats = pandas.DataFrame(columns=["area", "med_dem_dif"],
                                 index=range(1, len(land_bins), 1))
    land_stats["area"] = (land.groupby(['category']).count() * highres_pix_area / 1000000).dem_diff
    land_stats["med_dem_dif"] = (land.groupby(['category']).median()).dem_diff
    land_stats["med_dem_dif"] = land_stats["med_dem_dif"].where(land_stats["area"] >= land_stats["area"].sum() / 1000)

    ice_stats_list.append(ice_stats)
    land_stats_list.append(land_stats)
    ice_bins_list.append(ice_bins)
    land_bins_list.append(land_bins)

heights_ice = []
heights_land = []

for l in ice_bins_list:
    tmp = l[1] - l[0]
    heights_ice.append(tmp)

for l in land_bins_list:
    tmp = l[1] - l[0]
    heights_land.append(tmp)


# Plot ice data
fig, ax = plt.subplots(3, 2)

ax[0, 0].set_title("RGI-3 Canada North", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0, 0].barh(ice_bins_list[0][1:], ice_stats_list[0]["area"],
              height=(heights_ice[0] - heights_ice[0]/10)*-1,
              align="edge", color="tab:blue")
ax[0, 0].set_ylim((ice_bins_list[0].min(), ice_bins_list[0].max()))
ax[0, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[0, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[0, 0].tick_params(axis='both', labelsize=6)

ax00_1 = ax[0, 0].twiny()
ax00_1.plot(ice_stats_list[0]["med_dem_dif"], ice_bins_list[0][1:]-heights_ice[0]/2, color="tab:red")
ax00_1.set_xlabel("COP-250 minus MAR DEM median difference (m)", color="tab:red", fontsize=6)
ax00_1.tick_params(axis='x', labelcolor="tab:red")
ax00_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax00_1.tick_params(axis='both', labelsize=6)

#
ax[0, 1].set_title("RGI-4 Canada South", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0, 1].barh(ice_bins_list[1][1:], ice_stats_list[1]["area"],
              height=(heights_ice[1] - heights_ice[1]/10)*-1,
              align="edge", color="tab:blue")
ax[0, 1].set_ylim((ice_bins_list[1].min(), ice_bins_list[1].max()))
ax[0, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[0, 1].tick_params(axis='both', labelsize=6)

ax01_1 = ax[0, 1].twiny()
ax01_1.plot(ice_stats_list[1]["med_dem_dif"], ice_bins_list[1][1:]-heights_ice[1]/2, color="tab:red")
ax01_1.set_xlabel("COP-250 minus MAR DEM median difference (m)", color="tab:red", fontsize=6)
ax01_1.tick_params(axis='x', labelcolor="tab:red")
ax01_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax01_1.tick_params(axis='both', labelsize=6)

#
ax[1, 0].set_title("RGI-5 Greenland", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1, 0].barh(ice_bins_list[2][1:], ice_stats_list[2]["area"],
              height=(heights_ice[2] - heights_ice[2]/10)*-1,
              align="edge", color="tab:blue")
ax[1, 0].set_ylim((ice_bins_list[2].min(), ice_bins_list[2].max()))
ax[1, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[1, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[1, 0].tick_params(axis='both', labelsize=6)

ax10_1 = ax[1, 0].twiny()
ax10_1.plot(ice_stats_list[2]["med_dem_dif"], ice_bins_list[2][1:]-heights_ice[2]/2, color="tab:red")
ax10_1.tick_params(axis='x', labelcolor="tab:red")
ax10_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax10_1.tick_params(axis='both', labelsize=6)

#
ax[1, 1].set_title("RGI-6 Iceland", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1, 1].barh(ice_bins_list[3][1:], ice_stats_list[3]["area"],
              height=(heights_ice[3] - heights_ice[3]/10)*-1,
              align="edge", color="tab:blue")
ax[1, 1].set_ylim((ice_bins_list[3].min(), ice_bins_list[3].max()))
ax[1, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[1, 1].tick_params(axis='both', labelsize=6)

ax11_1 = ax[1, 1].twiny()
ax11_1.plot(ice_stats_list[3]["med_dem_dif"], ice_bins_list[3][1:]-heights_ice[3]/2, color="tab:red")
ax11_1.tick_params(axis='x', labelcolor="tab:red")
ax11_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax11_1.tick_params(axis='both', labelsize=6)

#
ax[2, 0].set_title("RGI-7 Svalbard", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[2, 0].barh(ice_bins_list[4][1:], ice_stats_list[4]["area"],
              height=(heights_ice[4] - heights_ice[4]/10)*-1,
              align="edge", color="tab:blue")
ax[2, 0].set_ylim((ice_bins_list[4].min(), ice_bins_list[4].max()))
ax[2, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[2, 0].set_xlabel("Area (km^2)", color="tab:blue", fontsize=6)
ax[2, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[2, 0].tick_params(axis='both', labelsize=6)

ax20_1 = ax[2, 0].twiny()
ax20_1.plot(ice_stats_list[4]["med_dem_dif"], ice_bins_list[4][1:]-heights_ice[4]/2, color="tab:red")
ax20_1.tick_params(axis='x', labelcolor="tab:red")
ax20_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax20_1.tick_params(axis='both', labelsize=6)

#
ax[2, 1].set_title("RGI-9 Russian Arctic", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[2, 1].barh(ice_bins_list[5][1:], ice_stats_list[5]["area"],
              height=(heights_ice[5] - heights_ice[5]/10)*-1,
              align="edge", color="tab:blue")
ax[2, 1].set_ylim((ice_bins_list[5].min(), ice_bins_list[5].max()))
ax[2, 1].set_xlabel("Area (km^2)", color="tab:blue", fontsize=6)
ax[2, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[2, 1].tick_params(axis='both', labelsize=6)

ax21_1 = ax[2, 1].twiny()
ax21_1.plot(ice_stats_list[5]["med_dem_dif"], ice_bins_list[5][1:]-heights_ice[5]/2, color="tab:red")
ax21_1.tick_params(axis='x', labelcolor="tab:red")
ax21_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax21_1.tick_params(axis='both', labelsize=6)

cm = 1/2.54
fig.set_size_inches(16*cm, 20*cm)
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, "DemEval_Ice.tif"
    ), dpi=300)
plt.close(fig=fig)


# Plot land data
fig, ax = plt.subplots(3, 2)

ax[0, 0].set_title("RGI-3 Canada North", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0, 0].barh(land_bins_list[0][1:], land_stats_list[0]["area"],
              height=(heights_land[0] - heights_land[0]/10)*-1,
              align="edge", color="tab:blue")
ax[0, 0].set_ylim((land_bins_list[0].min(), land_bins_list[0].max()))
ax[0, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[0, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[0, 0].tick_params(axis='both', labelsize=6)

ax00_1 = ax[0, 0].twiny()
ax00_1.plot(land_stats_list[0]["med_dem_dif"], land_bins_list[0][1:]-heights_land[0]/2, color="tab:red")
ax00_1.set_xlabel("COP-250 minus MAR DEM median difference (m)", color="tab:red", fontsize=6)
ax00_1.tick_params(axis='x', labelcolor="tab:red")
ax00_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax00_1.tick_params(axis='both', labelsize=6)

#
ax[0, 1].set_title("RGI-4 Canada South", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0, 1].barh(land_bins_list[1][1:], land_stats_list[1]["area"],
              height=(heights_land[1] - heights_land[1]/10)*-1,
              align="edge", color="tab:blue")
ax[0, 1].set_ylim((land_bins_list[1].min(), land_bins_list[1].max()))
ax[0, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[0, 1].tick_params(axis='both', labelsize=6)

ax01_1 = ax[0, 1].twiny()
ax01_1.plot(land_stats_list[1]["med_dem_dif"], land_bins_list[1][1:]-heights_land[1]/2, color="tab:red")
ax01_1.set_xlabel("COP-250 minus MAR DEM median difference (m)", color="tab:red", fontsize=6)
ax01_1.tick_params(axis='x', labelcolor="tab:red")
ax01_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax01_1.tick_params(axis='both', labelsize=6)

#
ax[1, 0].set_title("RGI-5 Greenland", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1, 0].barh(land_bins_list[2][1:], land_stats_list[2]["area"],
              height=(heights_land[2] - heights_land[2]/10)*-1,
              align="edge", color="tab:blue")
ax[1, 0].set_ylim((land_bins_list[2].min(), land_bins_list[2].max()))
ax[1, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[1, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[1, 0].tick_params(axis='both', labelsize=6)

ax10_1 = ax[1, 0].twiny()
ax10_1.plot(land_stats_list[2]["med_dem_dif"], land_bins_list[2][1:]-heights_land[2]/2, color="tab:red")
ax10_1.tick_params(axis='x', labelcolor="tab:red")
ax10_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax10_1.tick_params(axis='both', labelsize=6)

#
ax[1, 1].set_title("RGI-6 Iceland", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1, 1].barh(land_bins_list[3][1:], land_stats_list[3]["area"],
              height=(heights_land[3] - heights_land[3]/10)*-1,
              align="edge", color="tab:blue")
ax[1, 1].set_ylim((land_bins_list[3].min(), land_bins_list[3].max()))
ax[1, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[1, 1].tick_params(axis='both', labelsize=6)

ax11_1 = ax[1, 1].twiny()
ax11_1.plot(land_stats_list[3]["med_dem_dif"], land_bins_list[3][1:]-heights_land[3]/2, color="tab:red")
ax11_1.tick_params(axis='x', labelcolor="tab:red")
ax11_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax11_1.tick_params(axis='both', labelsize=6)

#
ax[2, 0].set_title("RGI-7 Svalbard", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[2, 0].barh(land_bins_list[4][1:], land_stats_list[4]["area"],
              height=(heights_land[4] - heights_land[4]/10)*-1,
              align="edge", color="tab:blue")
ax[2, 0].set_ylim((land_bins_list[4].min(), land_bins_list[4].max()))
ax[2, 0].set_ylabel("MAR elevation (m)", color="k", fontsize=6)
ax[2, 0].set_xlabel("Area (km^2)", color="tab:blue", fontsize=6)
ax[2, 0].tick_params(axis='x', labelcolor="tab:blue")
ax[2, 0].tick_params(axis='both', labelsize=6)

ax20_1 = ax[2, 0].twiny()
ax20_1.plot(land_stats_list[4]["med_dem_dif"], land_bins_list[4][1:]-heights_land[4]/2, color="tab:red")
ax20_1.tick_params(axis='x', labelcolor="tab:red")
ax20_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax20_1.tick_params(axis='both', labelsize=6)

#
ax[2, 1].set_title("RGI-9 Russian Arctic", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[2, 1].barh(land_bins_list[5][1:], land_stats_list[5]["area"],
              height=(heights_land[5] - heights_land[5]/10)*-1,
              align="edge", color="tab:blue")
ax[2, 1].set_ylim((land_bins_list[5].min(), land_bins_list[5].max()))
ax[2, 1].set_xlabel("Area (km^2)", color="tab:blue", fontsize=6)
ax[2, 1].tick_params(axis='x', labelcolor="tab:blue")
ax[2, 1].tick_params(axis='both', labelsize=6)

ax21_1 = ax[2, 1].twiny()
ax21_1.plot(land_stats_list[5]["med_dem_dif"], land_bins_list[5][1:]-heights_land[5]/2, color="tab:red")
ax21_1.tick_params(axis='x', labelcolor="tab:red")
ax21_1.axvline(x=0, color="tab:red", linestyle="--", linewidth=0.5)
ax21_1.tick_params(axis='both', labelsize=6)

cm = 1/2.54
fig.set_size_inches(16*cm, 20*cm)
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, "DemEval_Tundra.tif"
    ), dpi=300)
plt.close(fig=fig)
