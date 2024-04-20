# Import modules
import os
import pandas
import xarray
import rasterio
from rasterio.enums import Resampling
import numpy
import geopandas
import matplotlib.pyplot as plt

# Set up namespace
working_dir = "C:\postdoc_bristol\gis\Regions"  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"
output_dir = os.path.join(working_dir, "Plots")

if not os.path.lexists(output_dir):
    os.mkdir(output_dir)

domains = [
    "RGI_3_CanadaN",
    "RGI_4_CanadaS",
    "RGI_5_Greenland",
    "RGI_6_Iceland",
    "RGI_7_Svalbard"
]


# Import our data
domain_data = []
mar_domains = []

for name in domains:
    data_path = os.path.join(
        working_dir, name, "Results", name + "_TotalRunoff.txt")

    domain_path = os.path.join(
        working_dir, name, "selectors", name + "_MARextent_laeaa.shp")

    domain_data.append(
        pandas.read_csv(data_path))

    mar_domains.append(
        geopandas.read_file(domain_path))

mar_greenland = mar_domains[2]
mar_nongreenland = geopandas.GeoDataFrame(
    pandas.concat(mar_domains[0:2]+mar_domains[3:], ignore_index=True), crs=mar_domains[0].crs)

# Pre-process our data
for i, name in enumerate(domains):
    site_data = domain_data[i].loc[:, ["Downscaled ice runoff (m^3)",
                           "Downscaled ice runoff below snowline (m^3)",
                           "Downscaled land runoff (m^3)"]]

    site_data.index = pandas.to_datetime(domain_data[i].loc[:, "Date"])
    site_data.columns = ["RU_ice", "RU_ice_bsl", "RU_tundra"]

    site_data = site_data.groupby(pandas.Grouper(freq="Y")).sum() / 10 ** 9
    site_data.index = site_data.index.strftime("%Y").astype(int)

    if i == 0:
        gic = site_data

    if name == "RGI_5_Greenland":
        gris = site_data
    elif i == 0:
        gic = site_data
    else:
        gic = gic + site_data

summary_data = pandas.DataFrame(
    {
        "ice_nonGr": gic["RU_ice"],
        "ice_Gr": gris["RU_ice"],
        "tundra_nonGr": gic["RU_tundra"],
        "tundra_Gr": gris["RU_tundra"]
    },
    index=gic.index
)

del site_data, domain_data, gic, gris

summary_data_movmean = summary_data.rolling(window=5, min_periods=1, center=True).mean()
summary_data_movstd = summary_data.rolling(window=5, min_periods=1, center=True).std()

# Import Bamber 2018 FWF data
jgr_data = xarray.open_dataset(os.path.join(working_dir, "JGR_2018_data", "FWF17.v3_a.nc"))
jgr_data = jgr_data.rio.write_crs("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +k=1 +datum=WGS84 +units=m +no_defs")

mar_greenland = mar_greenland.to_crs(jgr_data.rio.crs)
mar_nongreenland = mar_nongreenland.to_crs(jgr_data.rio.crs)

mar_greenland = mar_greenland.buffer(distance=5000)
mar_nongreenland = mar_nongreenland.buffer(distance=5000)

jgr_data_gr = jgr_data.rio.clip(mar_greenland.geometry.values, all_touched=True)
jgr_data_nongr = jgr_data.rio.clip(mar_nongreenland.geometry.values, all_touched=True)

del jgr_data

jgr_gic = pandas.DataFrame(
    xarray.where(jgr_data_nongr["LSMGr"] == 1, numpy.nan, jgr_data_nongr["runoff_ice"]).sum(dim=["X", "Y"]),
    columns=["data"],
    index=pandas.to_datetime(jgr_data_nongr["TIME"])
)

jgr_gris = pandas.DataFrame(
    xarray.where(jgr_data_gr["LSMGr"] == 1, jgr_data_gr["runoff_ice"], numpy.nan).sum(dim=["X", "Y"]),
    columns=["data"],
    index=pandas.to_datetime(jgr_data_gr["TIME"])
)

jgr_gic_tundra = pandas.DataFrame(
    xarray.where(jgr_data_nongr["LSMGr"] == 1, numpy.nan, jgr_data_nongr["runoff_tundra"]).sum(dim=["X", "Y"]),
    columns=["data"],
    index=pandas.to_datetime(jgr_data_nongr["TIME"])
)

jgr_gris_tundra = pandas.DataFrame(
    xarray.where(jgr_data_gr["LSMGr"] == 1, jgr_data_gr["runoff_tundra"], numpy.nan).sum(dim=["X", "Y"]),
    columns=["data"],
    index=pandas.to_datetime(jgr_data_gr["TIME"])
)

del jgr_data_gr, jgr_data_nongr

jgr_gic = jgr_gic.groupby(pandas.Grouper(freq="Y")).sum()
jgr_gic.index = jgr_gic.index.strftime("%Y").astype(int)

jgr_gris = jgr_gris.groupby(pandas.Grouper(freq="Y")).sum()
jgr_gris.index = jgr_gris.index.strftime("%Y").astype(int)

jgr_gic_tundra = jgr_gic_tundra.groupby(pandas.Grouper(freq="Y")).sum()
jgr_gic_tundra.index = jgr_gic_tundra.index.strftime("%Y").astype(int)

jgr_gris_tundra = jgr_gris_tundra.groupby(pandas.Grouper(freq="Y")).sum()
jgr_gris_tundra.index = jgr_gris_tundra.index.strftime("%Y").astype(int)

jgr_summary_data = pandas.DataFrame(
    {
        "ice_nonGr": jgr_gic["data"],
        "ice_Gr": jgr_gris["data"],
        "tundra_nonGr": jgr_gic_tundra["data"],
        "tundra_Gr": jgr_gris_tundra["data"]
    },
    index=jgr_gic.index
)

del jgr_gic, jgr_gris, jgr_gic_tundra, jgr_gris_tundra

jgr_summary_data_movmean = jgr_summary_data.rolling(window=5, min_periods=1, center=True).mean()
jgr_summary_data_movstd = jgr_summary_data.rolling(window=5, min_periods=1, center=True).std()


# Import Mankoff 2020 data
man_data = xarray.open_dataset(os.path.join(working_dir, "Mankoff_2020_data", "MAR_ice.nc"))
man_ice_data = man_data["discharge"].sum(dim=["station"])
del man_data

period_sec = man_ice_data.indexes["time"].to_series().diff().dt.total_seconds().values
period_sec[0] = period_sec[1]
man_ice_data = man_ice_data * period_sec

man_ice_data = pandas.DataFrame(
    man_ice_data,
    columns=["data"],
    index=pandas.to_datetime(man_ice_data["time"])
)

man_ice_data = man_ice_data.groupby(pandas.Grouper(freq="Y")).sum() / 10 ** 9
man_ice_data.index = man_ice_data.index.strftime("%Y").astype(int)


#
man_data = xarray.open_dataset(os.path.join(working_dir, "Mankoff_2020_data", "MAR_land.nc"))
man_land_data = man_data["discharge"].sum(dim=["station"])
del man_data

period_sec = man_land_data.indexes["time"].to_series().diff().dt.total_seconds().values
period_sec[0] = period_sec[1]
man_land_data = man_land_data * period_sec

man_land_data = pandas.DataFrame(
    man_land_data,
    columns=["data"],
    index=pandas.to_datetime(man_land_data["time"])
)

man_land_data = man_land_data.groupby(pandas.Grouper(freq="Y")).sum() / 10 ** 9
man_land_data.index = man_land_data.index.strftime("%Y").astype(int)

man_summary_data = pandas.DataFrame(
    {
        "ice_Gr": man_ice_data["data"],
        "tundra_Gr": man_land_data["data"]
    },
    index=man_land_data.index
)

man_summary_data_movmean = man_summary_data.rolling(window=5, min_periods=1, center=True).mean()
man_summary_data_movstd = man_summary_data.rolling(window=5, min_periods=1, center=True).std()


#
comp_gic = pandas.DataFrame(
    {
        "this study": summary_data["ice_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data["ice_nonGr"]
    }
)

comp_gris = pandas.DataFrame(
    {
        "this study": summary_data["ice_Gr"],
        "Bamber et al. (2018)": jgr_summary_data["ice_Gr"],
        "Mankoff et al. (2020)": man_summary_data["ice_Gr"]
    }
)

comp_gic_tundra = pandas.DataFrame(
    {
        "this study": summary_data["tundra_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data["tundra_nonGr"]
    }
)

comp_gris_tundra = pandas.DataFrame(
    {
        "this study": summary_data["tundra_Gr"],
        "Bamber et al. (2018)": jgr_summary_data["tundra_Gr"],
        "Mankoff et al. (2020)": man_summary_data["tundra_Gr"]
    }
)

comp_gic.to_excel(os.path.join(output_dir, "Comparison_Ice_nonGR.xlsx"))
comp_gris.to_excel(os.path.join(output_dir, "Comparison_Ice_GR.xlsx"))
comp_gic_tundra.to_excel(os.path.join(output_dir, "Comparison_Tundra_nonGR.xlsx"))
comp_gris_tundra.to_excel(os.path.join(output_dir, "Comparison_Tundra_GR.xlsx"))

# Construct comparison dataframes
comp_gic_movmean = pandas.DataFrame(
    {
        "this study": summary_data_movmean["ice_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data_movmean["ice_nonGr"]
    }
)

comp_gris_movmean = pandas.DataFrame(
    {
        "this study": summary_data_movmean["ice_Gr"],
        "Bamber et al. (2018)": jgr_summary_data_movmean["ice_Gr"],
        "Mankoff et al. (2020)": man_summary_data_movmean["ice_Gr"]
    }
)

comp_gic_tundra_movmean = pandas.DataFrame(
    {
        "this study": summary_data_movmean["tundra_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data_movmean["tundra_nonGr"]
    }
)

comp_gris_tundra_movmean = pandas.DataFrame(
    {
        "this study": summary_data_movmean["tundra_Gr"],
        "Bamber et al. (2018)": jgr_summary_data_movmean["tundra_Gr"],
        "Mankoff et al. (2020)": man_summary_data_movmean["tundra_Gr"]
    }
)

comp_gic_movmean.to_excel(os.path.join(output_dir, "Comparison_MovMean_Ice_nonGR.xlsx"))
comp_gris_movmean.to_excel(os.path.join(output_dir, "Comparison_MovMean_Ice_GR.xlsx"))
comp_gic_tundra_movmean.to_excel(os.path.join(output_dir, "Comparison_MovMean_Tundra_nonGR.xlsx"))
comp_gris_tundra_movmean.to_excel(os.path.join(output_dir, "Comparison_MovMean_Tundra_GR.xlsx"))

# Construct comparison dataframes
comp_gic_movstd = pandas.DataFrame(
    {
        "this study": summary_data_movstd["ice_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data_movstd["ice_nonGr"]
    }
)

comp_gris_movstd = pandas.DataFrame(
    {
        "this study": summary_data_movstd["ice_Gr"],
        "Bamber et al. (2018)": jgr_summary_data_movstd["ice_Gr"],
        "Mankoff et al. (2020)": man_summary_data_movstd["ice_Gr"]
    }
)

comp_gic_tundra_movstd = pandas.DataFrame(
    {
        "this study": summary_data_movstd["tundra_nonGr"],
        "Bamber et al. (2018)": jgr_summary_data_movstd["tundra_nonGr"]
    }
)

comp_gris_tundra_movstd = pandas.DataFrame(
    {
        "this study": summary_data_movstd["tundra_Gr"],
        "Bamber et al. (2018)": jgr_summary_data_movstd["tundra_Gr"],
        "Mankoff et al. (2020)": man_summary_data_movstd["tundra_Gr"]
    }
)

comp_gic_movstd.to_excel(os.path.join(output_dir, "Comparison_MovStd_Ice_nonGR.xlsx"))
comp_gris_movstd.to_excel(os.path.join(output_dir, "Comparison_MovStd_Ice_GR.xlsx"))
comp_gic_tundra_movstd.to_excel(os.path.join(output_dir, "Comparison_MovStd_Tundra_nonGR.xlsx"))
comp_gris_tundra_movstd.to_excel(os.path.join(output_dir, "Comparison_MovStd_Tundra_GR.xlsx"))



#
import os
import pandas
import matplotlib.pyplot as plt

working_dir = "C:\postdoc_bristol\gis\Regions"  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"
output_dir = os.path.join(working_dir, "Plots")

comp_gic_movmean = pandas.read_excel(os.path.join(output_dir, "Comparison_MovMean_Ice_nonGR.xlsx"), index_col=0)
comp_gris_movmean = pandas.read_excel(os.path.join(output_dir, "Comparison_MovMean_Ice_GR.xlsx"), index_col=0)
comp_gic_tundra_movmean = pandas.read_excel(os.path.join(output_dir, "Comparison_MovMean_Tundra_nonGR.xlsx"), index_col=0)
comp_gris_tundra_movmean = pandas.read_excel(os.path.join(output_dir, "Comparison_MovMean_Tundra_GR.xlsx"), index_col=0)

comp_gic_movstd = pandas.read_excel(os.path.join(output_dir, "Comparison_MovStd_Ice_nonGR.xlsx"), index_col=0)
comp_gris_movstd = pandas.read_excel(os.path.join(output_dir, "Comparison_MovStd_Ice_GR.xlsx"), index_col=0)
comp_gic_tundra_movstd = pandas.read_excel(os.path.join(output_dir, "Comparison_MovStd_Tundra_nonGR.xlsx"), index_col=0)
comp_gris_tundra_movstd = pandas.read_excel(os.path.join(output_dir, "Comparison_MovStd_Tundra_GR.xlsx"), index_col=0)

#
fig, ax = plt.subplots(2, 2)
plt.rcParams.update({'font.size': 6})

l1, = ax[0,0].plot(comp_gris_movmean.index, comp_gris_movmean["this study"], color='r')
ax[0,0].fill_between(comp_gris_movmean.index, (comp_gris_movmean["this study"]-comp_gris_movstd["this study"]), (comp_gris_movmean["this study"]+comp_gris_movstd["this study"]), color='r', alpha=.1)

l2, = ax[0,0].plot(comp_gris_movmean.index, comp_gris_movmean["Bamber et al. (2018)"], color='b')
ax[0,0].fill_between(comp_gris_movmean.index, (comp_gris_movmean["Bamber et al. (2018)"]-comp_gris_movstd["Bamber et al. (2018)"]), (comp_gris_movmean["Bamber et al. (2018)"]+comp_gris_movstd["Bamber et al. (2018)"]), color='b', alpha=.1)

l3, = ax[0,0].plot(comp_gris_movmean.index, comp_gris_movmean["Mankoff et al. (2020)"], color='g')
ax[0,0].fill_between(comp_gris_movmean.index, (comp_gris_movmean["Mankoff et al. (2020)"]-comp_gris_movstd["Mankoff et al. (2020)"]), (comp_gris_movmean["Mankoff et al. (2020)"]+comp_gris_movstd["Mankoff et al. (2020)"]), color='g', alpha=.1)

ax[0,0].set_title("Greenland ice", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0,0].get_xaxis().set_ticks([])
ax[0,0].set_ylabel("Runoff (Gt/yr)", fontdict={'fontsize': 8})

#
ax[0,1].plot(comp_gris_tundra_movmean.index, comp_gris_tundra_movmean["this study"], color='r')
ax[0,1].fill_between(comp_gris_tundra_movmean.index, (comp_gris_tundra_movmean["this study"]-comp_gris_tundra_movstd["this study"]), (comp_gris_tundra_movmean["this study"]+comp_gris_tundra_movstd["this study"]), color='r', alpha=.1)

ax[0,1].plot(comp_gris_tundra_movmean.index, comp_gris_tundra_movmean["Bamber et al. (2018)"], color='b')
ax[0,1].fill_between(comp_gris_tundra_movmean.index, (comp_gris_tundra_movmean["Bamber et al. (2018)"]-comp_gris_tundra_movstd["Bamber et al. (2018)"]), (comp_gris_tundra_movmean["Bamber et al. (2018)"]+comp_gris_tundra_movstd["Bamber et al. (2018)"]), color='b', alpha=.1)

ax[0,1].plot(comp_gris_tundra_movmean.index, comp_gris_tundra_movmean["Mankoff et al. (2020)"], color='g')
ax[0,1].fill_between(comp_gris_tundra_movmean.index, (comp_gris_tundra_movmean["Mankoff et al. (2020)"]-comp_gris_tundra_movstd["Mankoff et al. (2020)"]), (comp_gris_tundra_movmean["Mankoff et al. (2020)"]+comp_gris_tundra_movstd["Mankoff et al. (2020)"]), color='g', alpha=.1)

ax[0,1].set_title("Greenland tundra", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[0,1].get_xaxis().set_ticks([])

#
ax[1,0].plot(comp_gic_movmean.index, comp_gic_movmean["this study"], color='r')
ax[1,0].fill_between(comp_gic_movmean.index, (comp_gic_movmean["this study"]-comp_gic_movstd["this study"]), (comp_gic_movmean["this study"]+comp_gic_movstd["this study"]), color='r', alpha=.1)

ax[1,0].plot(comp_gic_movmean.index, comp_gic_movmean["Bamber et al. (2018)"], color='b')
ax[1,0].fill_between(comp_gic_movmean.index, (comp_gic_movmean["Bamber et al. (2018)"]-comp_gic_movstd["Bamber et al. (2018)"]), (comp_gic_movmean["Bamber et al. (2018)"]+comp_gic_movstd["Bamber et al. (2018)"]), color='b', alpha=.1)

ax[1,0].set_title("non-Greenland ice", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1,0].set_xlabel("Date", fontdict={'fontsize': 8})
ax[1,0].set_ylabel("Runoff (Gt/yr)", fontdict={'fontsize': 8})

#
ax[1,1].plot(comp_gic_tundra_movmean.index, comp_gic_tundra_movmean["this study"], color='r')
ax[1,1].fill_between(comp_gic_tundra_movmean.index, (comp_gic_tundra_movmean["this study"]-comp_gic_tundra_movstd["this study"]), (comp_gic_tundra_movmean["this study"]+comp_gic_tundra_movstd["this study"]), color='r', alpha=.1)

ax[1,1].plot(comp_gic_tundra_movmean.index, comp_gic_tundra_movmean["Bamber et al. (2018)"], color='b')
ax[1,1].fill_between(comp_gic_tundra_movmean.index, (comp_gic_tundra_movmean["Bamber et al. (2018)"]-comp_gic_tundra_movstd["Bamber et al. (2018)"]), (comp_gic_tundra_movmean["Bamber et al. (2018)"]+comp_gic_tundra_movstd["Bamber et al. (2018)"]), color='b', alpha=.1)

ax[1,1].set_title("non-Greenland tundra", fontdict={'fontsize': 8, 'weight': 'bold'})
ax[1,1].set_xlabel("Date", fontdict={'fontsize': 8})

ax[0,0].legend([l1, l2, l3], ["this study", "Bamber et al. (2018)", "Mankoff et al. (2020)"], fontsize=6,
              loc="upper left")

cm = 1/2.54
fig.set_size_inches(16*cm, 14*cm)
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, "Comparison.tif"
    ), dpi=600)
plt.close(fig=fig)
