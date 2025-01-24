# Import modules
import os
import pandas
import xarray
import rasterio
from rasterio.enums import Resampling
import numpy
import geopandas
import matplotlib
import matplotlib.pyplot as plt

# Set up namespace
working_dir = "C:\\postdoc_bristol\\gis\\Regions"  # os.getenv("WORKING_DIR") "C:\postdoc_bristol\gis\Regions"
output_dir = os.path.join(working_dir, "Plots")

if not os.path.lexists(output_dir):
    os.mkdir(output_dir)

domains = [
    "RGI_3_CanadaN",
    "RGI_4_CanadaS",
    "RGI_5_Greenland",
    "RGI_6_Iceland",
    "RGI_7_Svalbard",
    "RGI_9_RussiaN"
]

domain_index = range(0, len(domains))

# Import our data
domain_data = []
mar_domains = []

for name in domains:
    data_path = os.path.join(
        working_dir, name, "Results", name + "_TotalRunoff.txt")

    domain_data.append(
        pandas.read_csv(data_path))

arctic = []
gris = []
gic = []

rmsd_ice = []
nrmsd_ice = []
avg_diff_ice = []
avg_reldiff_ice = []

rmsd_tundra = []
nrmsd_tundra = []
avg_diff_tundra = []
avg_reldiff_tundra = []

# Pre-process our data
for i, name in enumerate(domains):
    site_data = domain_data[i].loc[:, [
        "Raw ice runoff (m^3)",
        "Downscaled ice runoff (m^3)",
        "Raw land runoff (m^3)",
        "Downscaled land runoff (m^3)"
        ]
    ]

    site_data.index = pandas.to_datetime(domain_data[i].loc[:, "Date"])
    site_data.columns = ["RU_MAR_ice", "RU_ice", "RU_MAR_tundra", "RU_tundra"]

    site_data = site_data.groupby(pandas.Grouper(freq="YE")).sum() / 10 ** 9
    site_data.index = site_data.index.strftime("%Y").astype(int)

    site_data.index = [domain_index[i]] * site_data.shape[0]

    # Calculate stats, RMSD, NRSMD, and average difference
    rmsd_ice.append(
        ((site_data["RU_MAR_ice"] - site_data["RU_ice"]) ** 2).mean() ** (1/2)
    )
    nrmsd_ice.append(
        ((site_data["RU_MAR_ice"] - site_data["RU_ice"]) ** 2).mean() ** (1/2) / site_data["RU_MAR_ice"].mean() * 100
    )

    avg_diff_ice.append(
        (site_data["RU_MAR_ice"] - site_data["RU_ice"]).mean()
    )
    avg_reldiff_ice.append(
        ((site_data["RU_MAR_ice"] - site_data["RU_ice"]) / site_data["RU_MAR_ice"] * 100).mean()
    )

    rmsd_tundra.append(
        ((site_data["RU_MAR_tundra"] - site_data["RU_tundra"]) ** 2).mean() ** (1/2)
    )
    nrmsd_tundra.append(
        ((site_data["RU_MAR_tundra"] - site_data["RU_tundra"]) ** 2).mean() ** (1/2) / site_data["RU_MAR_tundra"].mean() * 100
    )

    avg_diff_tundra.append(
        (site_data["RU_MAR_tundra"] - site_data["RU_tundra"]).mean()
    )
    avg_reldiff_tundra.append(
        ((site_data["RU_MAR_tundra"] - site_data["RU_tundra"]) / site_data["RU_MAR_tundra"] * 100).mean()
    )

    arctic.append(site_data)

    if name == "RGI_5_Greenland":
        gris.append(site_data)
    else:
        gic.append(site_data)

# Concatenate the pre-processed data into single containers
arctic = pandas.concat(arctic, axis=0)
gic = pandas.concat(gic, axis=0)
gris = pandas.concat(gris, axis=0)

stats_ice = pandas.DataFrame(
    data={
        "rmsd": rmsd_ice,
        "nrmsd": nrmsd_ice,
        "avg_diff": avg_diff_ice,
        "avg_reldiff": avg_reldiff_ice,
    },
    index=domains
)

stats_tundra = pandas.DataFrame(
    data={
        "rmsd": rmsd_tundra,
        "nrmsd": nrmsd_tundra,
        "avg_diff": avg_diff_tundra,
        "avg_reldiff": avg_reldiff_tundra,
    },
    index=domains
)

# Save the statistics
stats_ice.to_csv(os.path.join(output_dir, "DownscalingEval_Ice.csv"))
stats_tundra.to_csv(os.path.join(output_dir, "DownscalingEval_Tundra.csv"))

# Create legend labels
domains = [
    "RGI-3 Canada North",
    "RGI-4 Canada South",
    "RGI-5 Greenland",
    "RGI-6 Iceland",
    "RGI-7 Svalbard",
    "RGI-9 Russian Arctic"
]

# Create "best-fit" line
line_ice = 1 * arctic["RU_MAR_ice"] + 0
line_tundra = 1 * arctic["RU_MAR_tundra"] + 0

# Plot the data
fig, ax = plt.subplots(1, 2)
plt.rcParams.update({'font.size': 6})

ax[0].plot(arctic["RU_MAR_ice"], line_ice, color='black', linewidth=0.5)
sc = ax[0].scatter(arctic["RU_MAR_ice"], arctic["RU_ice"],
                 c=arctic.index, cmap="Dark2", marker=".", s=5)
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[0].set_title("Ice runoff", fontdict={'fontsize': 10, 'weight': 'bold'})
ax[0].set_xlabel("MAR (Gt/yr)", fontdict={'fontsize': 8})
ax[0].set_ylabel("Downscaled (Gt/yr)", fontdict={'fontsize': 8})

handles, labels = sc.legend_elements()
legend = ax[0].legend(handles, domains, loc="upper left", title="Regions", fontsize=6)


ax[1].plot(arctic["RU_MAR_tundra"], line_tundra, color='black', linewidth=0.5)
sc2 = ax[1].scatter(arctic["RU_MAR_tundra"], arctic["RU_tundra"],
                 c=arctic.index, cmap="Dark2", marker=".", s=5)
ax[1].set_xscale("log")
ax[1].set_yscale("log")

ax[1].set_title("Tundra runoff", fontdict={'fontsize': 10, 'weight': 'bold'})
ax[1].set_xlabel("MAR (Gt/yr)", fontdict={'fontsize': 8})

cm = 1/2.54
fig.set_size_inches(16*cm, 8*cm)
fig.tight_layout(pad=1)
fig.savefig(os.path.join(
        output_dir, "DownscalingEval.tif"
    ), dpi=300)
plt.close(fig=fig)
