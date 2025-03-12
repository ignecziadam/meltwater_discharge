# This script contains code from Mankoff et al. (2020)
# https://github.com/GEUS-Glaciology-and-Climate/freshwater/blob/main/lob.org
# Import modules
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import os

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

# Define a custom axis alignment tool
# http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
def adjust_spines(ax,spines, offset=10):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', offset)) # outward
            # by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't
            # draw spine

        # turn off ticks where there
        # is no spine
        if 'left' in spines:
            ax.yaxis.set_tick_params(length=5)
            ax.yaxis.set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
        elif 'right' in spines:
            ax.yaxis.set_tick_params(length=5)
            ax.yaxis.set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_tick_params(length=5)
            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_label_position('bottom')
        elif 'top' in spines:
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_tick_params(length=5)
            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_label_position('top')
        else:
            # no xaxis
            # ticks
            ax.xaxis.set_ticks([])


# set up data paths
working_dir = "C:\\postdoc_bristol\\gis\\Regions"
output_dir = os.path.join(working_dir, "Plots")

# Convert Watson data
w = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "raw", "Watson_River_discharge_daily.txt"),
                sep="\s+",
                parse_dates=[[0,1,2]],
                index_col=0)\
      .drop(["DayOfYear", "DayOfCentury"], axis='columns')\
      .rename({"WaterFluxDiversOnly(m3/s)"         : "divers",
               "Uncertainty(m3/s)"                 : "divers_err",
               "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
               "Uncertainty(m3/s).1"               : "divers_t_err",
               "WaterFluxCumulative(km3)"          : "cum",
               "Uncertainty(km3)"                  : "cum_err"},
              axis='columns')

obs = w[['divers_t','divers_t_err']].rename({'divers_t':'Observed',
                                             'divers_t_err':'Observed error'}, axis='columns')
obs.index.name = 'time'
obs.to_csv(os.path.join(working_dir, "mankoff_discharge", "obs_W.csv"))

# Convert Qaanaaq data
obs = pd.read_csv(
    os.path.join(working_dir, "mankoff_discharge", "raw", "discharge2017.txt"),
    index_col=0, parse_dates=True)
tmp = pd.read_csv(
    os.path.join(working_dir, "mankoff_discharge", "raw", "discharge2018.txt"),
    index_col=0, parse_dates=True)
obs = pd.concat((obs,tmp))

tmp = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "raw", "discharge2019.txt"),
                  index_col=0, parse_dates=True)
obs = pd.concat((obs,tmp))

obs = obs.resample('1D')\
         .mean()\
         .rename({'Discharge':'Observed'}, axis='columns')

obs.index.name = "time"
obs.to_csv(os.path.join(working_dir, "mankoff_discharge", "obs_Q.csv"))

# Convert Narsarsuaq data
obs = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "raw", "NarsarsuaqDischarge2013.csv"))\
        .rename({"Q (m3 sec-1)" : "Observed"}, axis="columns")

obs.index = datetime.datetime(2013,1,1) + np.array([datetime.timedelta(_-1) for _ in obs['DecDay']])
obs.index.name = "time"
obs.drop('DecDay', inplace=True, axis='columns')
obs = obs.resample('1D').mean().dropna()

obs.to_csv(os.path.join(working_dir, "mankoff_discharge", "obs_Ks.csv"))

# GEM
obs = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "raw", "GEM.csv"),
                  parse_dates=True, index_col=0)
obs.index.name = 'time'

nloc = [['Kobbefjord', "Kb"],
        ['Oriartorfik', "O"],
        ['Teqinngalip', "T"],
        ['Kingigtorssuaq', "K"],
        ['Røde_Elv', "R"],
        ['Zackenberg', "Z"]]

for nl in nloc:
    obs[nl[0]].to_csv(
        os.path.join(working_dir, "mankoff_discharge", "obs_" + nl[1] + ".csv"))


# set up namespace (and the IDs of the corresponding drainage basins by investigating the location of river gauges and basin coverage)
names = ['G Kobbefjord & Oriartorfik',
         'Ks Kiattuut Sermiat',
         'Q Qaanaaq',
         'R Røde Elv',
         'W Watson',
         'Z Zackenberg']

basin_ID = {
    'G': 50847,
    'Ks': 60201,
    'Q': 9794,
    'R': 32818,
    'W': 41042,
    'Z': 17336}

name = [' '.join(_.split(" ")[1:]) for _ in names]
loc = [_.split(" ")[0] for _ in names]

# Import our downscaled coastal discharge data
domain_data = xr.open_dataset(
    os.path.join(working_dir, "RGI_5_Greenland", "Results", "RGI_5_Greenland_BasinRunoff.nc"))

obs = {} # store all in dict of dataarrays

# integrate the river gauge data with the downscaled coastal discharge estimations
for i, l in enumerate(loc):
    if l == 'G':
        tmp_Kb = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "obs_Kb.csv"),
                             index_col=0, parse_dates=True)
        tmp_O = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "obs_O.csv"),
                             index_col=0, parse_dates=True)
        tmp_Kb.columns = ['obs'] if l != 'W' else ['obs', 'err']
        tmp_O.columns = ['obs'] if l != 'W' else ['obs', 'err']
        df_obs = tmp_Kb + tmp_O
    else:
        df_obs = pd.read_csv(os.path.join(working_dir, "mankoff_discharge", "obs_" + l + ".csv"),
                             index_col=0, parse_dates=True)
        df_obs.columns = ['obs'] if l != 'W' else ['obs', 'err']

    MAR_ice = domain_data["IceRunoff"].sel(BasinID=basin_ID[l]).to_series()
    MAR_land = domain_data["LandRunoff"].sel(BasinID=basin_ID[l]).to_series()
    period_sec = MAR_ice.index.to_series().diff().dt.total_seconds().values
    period_sec[0] = period_sec[1]
    MAR = (MAR_ice + MAR_land) / period_sec

    MAR = pd.DataFrame({'MAR': MAR})
    MAR['MAR'] = MAR['MAR'].rolling('7D', min_periods=5).mean()

    df = df_obs.merge(MAR, left_index=True, right_index=True)

    df.attrs['name'] = name[i]
    obs[l] = df


# one entry with everything, no time index, just all observation and model points
o,MAR = [],[]
for k in loc:
    o = np.append(o, obs[k]['obs'])
    MAR = np.append(MAR, obs[k]['MAR'])
df = pd.DataFrame((o,MAR), index=['obs','MAR']).T
df.attrs['name'] = "all"
obs_all = df

# same as above but without GEM basins
o,MAR = [],[]
for k in loc:
    if k in ['G']: continue
    o = np.append(o, obs[k]['obs'])
    MAR = np.append(MAR, obs[k]['MAR'])
df = pd.DataFrame((o,MAR), index=['obs','MAR']).T
df.attrs['name'] = "noGEM"
obs_noGEM = df


# Plot the data
########################################
fig, ax = plt.subplots(1,2)

# Plot all basins alone
for k in obs.keys():
    df = obs[k]
    df = df.replace(0, np.nan).dropna()
    df = np.log10(df)

    df_mod = df.where((-3 < df) & (df < 4), np.nan).dropna()
    ax[0].scatter(df_mod['obs'], df_mod['MAR'], marker='.', alpha=0.1,
                label=df_mod.attrs['name'], edgecolor='none', clip_on=False)

# fit to all basins together
df = obs_all
df = np.log10(df)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

rmse = ((df['obs'] - df['MAR']) ** 2).mean() ** (1/2)
mbe = (df['obs'] - df['MAR']).mean()

df.sort_values(by='obs', inplace=True)
x = df['obs']
y_MAR = df['MAR']

X = sm.add_constant(x)
model = sm.OLS(y_MAR, X)
results = model.fit()
prstd, iv_l, iv_u = wls_prediction_std(results)
ax[0].fill_between(x, iv_u, iv_l, color="grey", alpha=0.25)
ax[0].text(0.75, 0.15, 'r$^{2}$:' + str(round(results.rsquared,2)), transform=ax[0].transAxes, horizontalalignment='left')
ax[0].text(0.75, 0.075, 'rmse:' + str(round(rmse,2)), transform=ax[0].transAxes, horizontalalignment='left')
ax[0].text(0.75, 0, 'mbe:' + str(round(mbe,2)), transform=ax[0].transAxes, horizontalalignment='left')

# repeat but without GEM basins
df = obs_noGEM
df = np.log10(df)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

rmse = ((df['obs'] - df['MAR']) ** 2).mean() ** (1/2)
mbe = (df['obs'] - df['MAR']).mean()

df.sort_values(by='obs', inplace=True)
x = df['obs']
y_MAR = df['MAR']

X = sm.add_constant(x)
model = sm.OLS(y_MAR, X)
results = model.fit()
prstd, iv_l, iv_u = wls_prediction_std(results)
ax[0].fill_between(x, iv_u, iv_l, color="red", alpha=0.1)
ax[0].text(0.75, 0.4, 'r$^{2}$:' + str(round(results.rsquared,2)), transform=ax[0].transAxes, horizontalalignment='left', color='red')
ax[0].text(0.75, 0.325, 'rmse:' + str(round(rmse,2)), transform=ax[0].transAxes, horizontalalignment='left', color='red')
ax[0].text(0.75, 0.25, 'mbe:' + str(round(mbe,2)), transform=ax[0].transAxes, horizontalalignment='left', color='red')

# Format the plot
coords = np.log10([1E-3, 1E4])

kw = {'alpha': 0.5, 'linewidth': 1, 'color': 'k', 'linestyle': '-'}
ax[0].plot(np.log10([1E-3, 1E4]), np.log10([1E-3, 1E4]), **kw)
ax[0].plot(np.log10([1E-3, 1E4]), np.log10([1E-3 / 5, 1E4 / 5]), **kw)
ax[0].plot(np.log10([1E-3, 1E4]), np.log10([1E-3 * 5, 1E4 * 5]), **kw)

ax[0].set_ylim([-3, 4])
ax[0].set_xlim([-3, 4])
ax[0].set_yticks([-3, -2, -1, 0, 1, 2, 3, 4])
ax[0].set_yticklabels(
    ['10$^{-3}$', '10$^{-2}$', '10$^{-1}$', '10$^{0}$', '10$^{1}$', '10$^{2}$', '10$^{3}$', '10$^{4}$'])
ax[0].set_xticks(ax[0].get_yticks())
ax[0].set_xticklabels(ax[0].get_yticklabels())

adjust_spines(ax[0], ['left', 'bottom'])

ax[0].set_xlabel('Observed (m^3/s)')
ax[0].set_ylabel('MAR (m^3/s)')

leg = ax[0].legend(fontsize=6, frameon=False, bbox_to_anchor=(0, 1), loc='upper right', mode="expand")

for lh in leg.legendHandles:
    lh.set_alpha(1)

plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)

mticks = np.array([np.log10(np.linspace(2 * _, 9 * _, num=8)) for _ in [0.001, 0.01, 0.1, 1, 10, 100, 1000]]).ravel()

ax[0].set_xticks(mticks, minor=True)
ax[0].set_yticks(mticks, minor=True)


###
# Plot annual data
###
for k in obs.keys():
    df = obs[k]
    name = df.attrs['name']
    df = df.replace(0, np.nan).dropna()
    # m^3/s summed by year -> km^3/yr
    df = df.resample('YE').sum() * 86400
    df = np.log10(df)
    ax[1].scatter(df['obs'], df['MAR'], marker='$\mathrm{'+k+'}$', alpha=0.9,
                label=name, clip_on=False, zorder=99)

# combine all into one for confidence intervals
# one entry with everything, no time index, just all observation and model points
o,MAR = [],[]
for k in obs.keys():
    o = np.append(o, obs[k]['obs'].resample('YE').sum())
    MAR = np.append(MAR, obs[k]['MAR'].resample('YE').sum())

# m^3/s -> m^3/yr
df = pd.DataFrame((o,MAR), index=['obs','MAR']).T * 86400
df = np.log10(df)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

rmse = ((df['obs'] - df['MAR']) ** 2).mean() ** (1/2)
mbe = (df['obs'] - df['MAR']).mean()

df.sort_values(by='obs', inplace=True)
x = df['obs']
y_MAR = df['MAR']

X = sm.add_constant(x)
model = sm.OLS(y_MAR, X)
results = model.fit()
prstd, iv_l, iv_u = wls_prediction_std(results)
ax[1].fill_between(x, iv_u, iv_l, color="grey", alpha=0.25)
ax[1].text(0.4, 0.15, 'r$^{2}$:' + str(round(results.rsquared,2)), transform=ax[1].transAxes, horizontalalignment='left')
ax[1].text(0.4, 0.075, 'rmse:' + str(round(rmse,2)), transform=ax[1].transAxes, horizontalalignment='left')
ax[1].text(0.4, 0, 'mbe:' + str(round(mbe,2)), transform=ax[1].transAxes, horizontalalignment='left')

ax[1].set_xlabel('Observed (m^3)')

kw = {'alpha':0.5, 'linewidth':1, 'color':'k', 'linestyle':'-'}
ax[1].plot(np.log10([1E6,1E10]), np.log10([1E6,1E10]), **kw)
ax[1].plot(np.log10([1E6,1E10]), np.log10([1E6/2,1E10/2]), **kw)
ax[1].plot(np.log10([1E6,1E10]), np.log10([1E6*2,1E10*2]), **kw)

ax[1].set_ylim([6,10])
ax[1].set_xlim([6,10])
ax[1].set_yticks([6,7,8,9,10])
ax[1].set_yticklabels(['10$^{6}$','10$^{7}$','10$^{8}$','10$^{9}$','10$^{10}$'])
ax[1].set_xticks(ax[1].get_yticks())
ax[1].set_xticklabels(ax[1].get_yticklabels())

adjust_spines(ax[1], ['left','bottom'])

ax[1].set_ylabel('MAR (m^3)')

leg = ax[1].legend(fontsize=6, frameon=False, bbox_to_anchor=(0, 1), loc='upper right', mode="expand")
ax[1].set_zorder(-2)
for lh in leg.legendHandles:
    lh.set_alpha(1)

for i,l in enumerate(leg.texts):
    l.set_y(-1.5)

#
cm = 1/2.54

fig.set_tight_layout(True)
fig.set_size_inches(8, 3.5)
fig.savefig(os.path.join(
        output_dir, "DischargeComp.tif"
    ), bbox_inches='tight', dpi=300)
plt.close(fig=fig)

