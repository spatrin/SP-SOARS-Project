# Evaluating the Representation of Fires in Georgia, USA, in the FINN Biomass Burning Emissions Inventory

## Research Questions

RQ1.
How accurately does FINN represent agricultural and prescribed fires in Georgia during June 2022?

RQ2.
To what extent do FINN detections spatially align with EPA-reported fire locations?

RQ3.
How accurately does FINN represent fire based on PM2.5 emissions?

RQ4.
How accurately does FINN represent fire based on burned area?

RQ5.
Is June 2022 representative of FINN performance across the 2022 fire season?

### 1. Setup


```python
import sys
import os

import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# from scipy.stats import ttest_ind, ks_2samp
from numba import njit, prange
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

project_root = "/glade/u/home/spatrin"
FIG_DIR = f"{project_root}/final_figures"
os.makedirs(FIG_DIR, exist_ok=True)

src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

print("Project root:", project_root)
```

    Project root: /glade/u/home/spatrin



```python
from utils import *
from data_loading import *
from pipeline import *
from spatial_utils import *
from geo_utils import *
from conversion import *
from aggregation import *
```


```python
FIRE_ORDER = FIRE_TYPES
ACRES_TO_KM2 = 0.00404686

print("Fire type order used throughout:", FIRE_ORDER)
```

    Fire type order used throughout: ['Agricultural', 'Prescribed', 'Wildfire']



```python
sns.set_theme(style="whitegrid", context="paper")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def savefig(fig, name):
    """Save a figure into FIG_DIR using a consistent naming convention."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=False)
    return path
```


```python
cmap = plt.cm.Reds

cmap_trim = mcolors.LinearSegmentedColormap.from_list(
    "Reds_trim",
    cmap(np.linspace(0.3, 1, 256))
)
```

### 2. Data Loading


```python
epa = load_epa_data()
epa_june = filter_june(epa)

gdf_june = gpd.GeoDataFrame(
    epa_june,
    geometry=gpd.points_from_xy(epa_june["longitude"], epa_june["latitude"]),
    crs="EPSG:4326"
)

print(f"Full-year EPA fires : {len(epa):,}")
print(f"June EPA fires      : {len(epa_june):,}")
```

    Full-year EPA fires : 79,563
    June EPA fires      : 2,771



```python
gdf_annual = gpd.GeoDataFrame(
    epa,
    geometry=gpd.points_from_xy(
        epa["longitude"],
        epa["latitude"]
    ),
    crs="EPSG:4326"
)
```


```python
# --- FINN: June 2022, gridded + converted to tons/day ---
finn_data = load_finn_june_processed()

lat = finn_data["lat"].values
lon = finn_data["lon"].values
pm_tons = finn_data["pm_tons"]     # (time, lat, lon), tons/day
pm_total = finn_data["pm_total"]   # (lat, lon), June total
pm_daily = finn_data["pm_daily"]   # (time,), June daily total

ga_mask = get_georgia_mask(lat, lon)

print("FINN grid shape (time, lat, lon):", pm_tons.shape)
print("Georgia grid cells:", int(np.sum(ga_mask)))
```

    /glade/u/home/spatrin/src/data_loading.py:42: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.
      ds = xr.open_dataset(


    FINN grid shape (time, lat, lon): (30, 1799, 3600)
    Georgia grid cells: 1466



```python
finn_annual = load_finn_annual_processed()

pm_annual_total = finn_annual["pm_total"]
annual_time = finn_annual["time"]
```

    /glade/u/home/spatrin/src/data_loading.py:42: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.
      ds = xr.open_dataset(



```python
pm_tons.sum
```




    <bound method DataArrayAggregations.sum of <xarray.DataArray 'fire_modisviirs_PM25' (time: 30, lat: 1799, lon: 3600)> Size: 777MB
    dask.array<mul, shape=(30, 1799, 3600), dtype=float32, chunksize=(1, 600, 1200), chunktype=numpy.ndarray>
    Coordinates:
      * time     (time) datetime64[ns] 240B 2022-06-01 2022-06-02 ... 2022-06-30
      * lat      (lat) float32 7kB -89.95 -89.85 -89.75 -89.65 ... 89.65 89.75 89.85
      * lon      (lon) float32 14kB -180.0 -179.8 -179.8 ... 179.8 179.9 180.0
    Attributes:
        units:      molecules/cm^2/s
        map:        PM25->PM25;aerosol
        long_name:  modisviirs_PM25 fire emissions>



### 3. Methods


```python
def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    n_x = len(x)
    n_y = len(y)

    gt = 0
    lt = 0

    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return (gt - lt) / (n_x * n_y)

def summarize_skewed(x):
    return pd.Series({
        "N": len(x),
        "Median": np.median(x),
        "Q1": np.percentile(x,25),
        "Q3": np.percentile(x,75),
        "P95": np.percentile(x,95),
        "Total": np.sum(x)
    })
```

#### 3.1 EPA

**Figure 1.**  EPA point fires and EPA emissions after aggregation to the FINN 0.1° grid.


```python
# --- Figure 1: EPA fire locations + gridded PM2.5 emissions ---
epa_grid_pm25 = epa_to_finn_grid(gdf_june, lat, lon, weight_type="pm25")
epa_grid_pm25 = apply_ga_mask(epa_grid_pm25, ga_mask)

plot_grid = epa_grid_pm25.copy()
plot_grid[plot_grid <= 0] = np.nan

cmap_reds = mcolors.LinearSegmentedColormap.from_list(
    "reds_trim", plt.cm.Reds(np.linspace(0.3, 1, 256))
)

vmin = np.nanpercentile(plot_grid, 1)
vmax = np.nanpercentile(plot_grid, 99)
norm = LogNorm(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
fig.subplots_adjust(left=0.05, right=0.9, bottom=0.08, top=0.87, wspace=0.02)

# Panel A: EPA point locations, sized by reported PM2.5
ax = axes[0]
basemap(ax)
vals = gdf_june["pm2.5"].fillna(0).values
vals_sqrt = np.sqrt(vals)
sizes = 5 + (vals_sqrt / (vals_sqrt.max() + 1e-6)) * 80
ax.scatter(
    gdf_june.geometry.x, gdf_june.geometry.y,
    s=sizes, color="#cb181d", alpha=0.6,
    transform=ccrs.PlateCarree(), zorder=4
)
ax.set_title("A. EPA Fire Locations", fontsize=12, pad=8, loc="left")

# Panel B: EPA gridded onto the FINN lattice
ax = axes[1]
basemap(ax)
mesh = ax.pcolormesh(
    lon, lat, plot_grid, cmap=cmap_reds, norm=norm,
    shading="nearest", zorder=4
)
cax = inset_axes(
    axes[1], width="4%", height="65%", loc="center right",
    bbox_to_anchor=(0.05, 0., 1, 1), bbox_transform=axes[1].transAxes, borderpad=0
)
cbar = fig.colorbar(mesh, cax=cax)
cbar.set_label("PM$_{2.5}$ (tons)", fontsize=11)
cbar.ax.tick_params(labelsize=10)
ax.set_title("B. EPA PM$_{2.5}$ Emissions Grid", fontsize=12, pad=8, loc="left")

fig.suptitle("EPA Fire Emissions in Georgia, June 2022", fontsize=16, fontweight="bold", y=0.98)

savefig(fig, "fig01_epa_pt_grid.png")
plt.show()
```


    
![png](output_17_0.png)
    



```python
epa_annual_summary = (
    epa
    .groupby("type")["pm2.5"]
    .apply(summarize_skewed)
)

display(epa_annual_summary)
```


    type                
    Agricultural  N         16704.000000
                  Median        0.056271
                  Q1            0.020478
                  Q3            0.204782
                  P95           0.819128
                  Total      3259.254103
    Prescribed    N         55926.000000
                  Median        0.262079
                  Q1            0.055368
                  Q3            1.332608
                  P95           6.265829
                  Total     98155.099630
    Wildfire      N          6933.000000
                  Median        0.038021
                  Q1            0.007801
                  Q3            0.189272
                  P95           1.667189
                  Total      4140.379911
    Name: pm2.5, dtype: float64


#### 3.2 Detection Classification


```python
def classify_all_days(gdf, finn, lat, lon):
    """Classify every EPA fire in `gdf` as matched/missed using the nearest-day FINN field."""
    results = []
    for date in sorted(gdf["date"].dt.normalize().unique()):
        epa_day = gdf[gdf["date"].dt.normalize() == date]
        if len(epa_day) == 0:
            continue
        finn_day = finn["pm_tons"].sel(time=date, method="nearest").values
        results.append(classify_missed_fires(epa_day, finn_day, lat, lon))
    return pd.concat(results, ignore_index=True)

fire_match_df = classify_all_days(gdf_june, finn_data, lat, lon)
fire_match_df["area_km2"] = fire_match_df["area"] * ACRES_TO_KM2
fire_match_df["Detection Status"] = fire_match_df["matched"].map({1: "Detected", 0: "Missed"})

print(f"Total fires classified : {len(fire_match_df):,}")
print(f"Detected               : {int(fire_match_df['matched'].sum()):,}")
print(f"Missed                 : {int((fire_match_df['matched'] == 0).sum()):,}")
```

    Total fires classified : 2,771
    Detected               : 260
    Missed                 : 2,511


#### 3.3 Detection Metrics

Discussion:
TP
FN
FP
Precision
Recall
F1
Jaccad
Dice


#### 3.4 Statistical Analyses

- mannwhitneyu  - 
kruska    l- 
cliffs_del     t- a
lo
   git

| Question                          | Test                |
| --------------------------------- | ------------------- |
| June vs annual representativeness | Mann–Whitney U      |
| Fire-type differences             | Kruskal–Wallis      |
| Detection \~ PM2.5                | Logistic regression |
| Detection \~ area                 | Logistic regression |


## 4. Results

### 4.1 EPA Fire Climatology in Georgia During 2022

Fire counts and seasonal trends across all of 2022, before narrowing to the June study perdio.


**Figure 2.** Total EPA fire events by type, full year 2022.


```python
counts_full = epa["type"].value_counts().reindex(FIRE_ORDER)
colors_full = [FIRE_COLORS[t] for t in FIRE_ORDER]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(counts_full.index, counts_full.values, color=colors_full, edgecolor="black")

ax.bar_label(
    ax.containers[0],
    padding=5
)
ax.set_ylim(0, counts_full.max() * 1.2)

ax.set_title("Number of Fire Events by Type in Georgia (2022)")
ax.set_xlabel("Fire Type")
ax.set_ylabel("Count")
sns.despine(ax=ax)

plt.tight_layout()
savefig(fig, "fig02_fire_counts_2022.png")
plt.show()
```


    
![png](output_29_0.png)
    


**Figure 3.** Monthly fire-count trends by type across 2022 (time series).


```python
monthly_counts = (
    epa.groupby(["month", "type"]).size().unstack().reindex(columns=FIRE_ORDER)
)

fig, ax = plt.subplots(figsize=(9, 5))
for t in FIRE_ORDER:
    ax.plot(monthly_counts.index, monthly_counts[t], label=t, color=FIRE_COLORS[t], linewidth=2)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTHS_LABELS, rotation=45)

ax.set_title("Monthly Fire Trends in Georgia (2022)")
ax.set_xlabel("Month")
ax.set_ylabel("Fire Count")
ax.grid(alpha=0.3)
ax.legend(title="Fire Type")
sns.despine(ax=ax)

plt.tight_layout()
savefig(fig, "fig03_monthly_fire_trends.png")
plt.show()
```


    
![png](output_31_0.png)
    


**Figure 4.** Monthly fire emission trends by type across 2022 (time series).


```python
monthly_pm25 = (
    epa.groupby(["month", "type"])["pm2.5"]
       .sum()
       .unstack()
       .reindex(columns=FIRE_ORDER)
)

fig, ax = plt.subplots(figsize=(9, 5))

for t in FIRE_ORDER:
    ax.plot(
        monthly_pm25.index,
        monthly_pm25[t],
        label=t,
        color=FIRE_COLORS[t],
        linewidth=2,
        marker="o"
    )

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTHS_LABELS, rotation=45)

ax.set_title("Monthly EPA PM$_{2.5}$ Emissions by Fire Type in Georgia (2022)")
ax.set_xlabel("Month")
ax.set_ylabel("PM$_{2.5}$ Emissions (tons)")
ax.legend(title="Fire Type")
ax.grid(alpha=0.3)

sns.despine(ax=ax)

plt.tight_layout()
savefig(fig, "fig04_monthly_pm25_by_type.png")
plt.show()
```


    
![png](output_33_0.png)
    



```python
annual_summary = (
    epa
    .groupby("type")["pm2.5"]
    .apply(summarize_skewed)
)

display(annual_summary)
```


    type                
    Agricultural  N         16704.000000
                  Median        0.056271
                  Q1            0.020478
                  Q3            0.204782
                  P95           0.819128
                  Total      3259.254103
    Prescribed    N         55926.000000
                  Median        0.262079
                  Q1            0.055368
                  Q3            1.332608
                  P95           6.265829
                  Total     98155.099630
    Wildfire      N          6933.000000
                  Median        0.038021
                  Q1            0.007801
                  Q3            0.189272
                  P95           1.667189
                  Total      4140.379911
    Name: pm2.5, dtype: float64



```python
ag = epa.loc[
    epa.type=="Agricultural",
    "pm2.5"
]

pr = epa.loc[
    epa.type=="Prescribed",
    "pm2.5"
]

wf = epa.loc[
    epa.type=="Wildfire",
    "pm2.5"
]

H,p = kruskal(
    ag,
    pr,
    wf
)
```


```python
H,p = kruskal(
    ag,
    pr,
    wf
)

print(
    f"Kruskal-Wallis H={H:.2f}, p={p:.3e}"
)

print("\nMedian PM2.5")

for t in FIRE_ORDER:

    vals = epa.loc[
        epa["type"]==t,
        "pm2.5"
    ]

    print(
        f"{t}: {np.median(vals):.2f}"
    )
```

    Kruskal-Wallis H=8112.85, p=0.000e+00
    
    Median PM2.5
    Agricultural: 0.06
    Prescribed: 0.26
    Wildfire: 0.04


### 4.2 Is June Representative of 2022?

**Figure .** PM$_{2.5}$ and burned-area distributions (violin + boxplot), split by detection status. Statistical tests below support the annotated $p$-values.


```python
fire_match_df["log_pm25"] = np.log10(fire_match_df["pm2.5"] + 0.01)
fire_match_df["log_area"] = np.log10(fire_match_df["area_km2"] + 1e-4)  # area in km^2

violin_color = "#9ecae1"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))


def violin_panel(ax, df, y_col, ylabel, panel_label, title, log_scale=False):
    sns.violinplot(data=df, x="Detection Status", y=y_col, inner=None,
                   color=violin_color, saturation=1, cut=0, linewidth=1.2, ax=ax)
    sns.boxplot(data=df, x="Detection Status", y=y_col, width=0.18, showcaps=True,
                fliersize=0,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="black", linewidth=1.2),
                capprops=dict(color="black", linewidth=1.2), ax=ax)
    sample = df.sample(min(800, len(df)), random_state=42)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top")
    # ax.text(0.5, 0.95, r"$Mann–Whitney p = 0.001 /n δ = 0.43$", transform=ax.transAxes, ha="center", fontsize=12)
    sns.despine(ax=ax)


violin_panel(axes[0], fire_match_df, "log_pm25",
             r"log$_{10}$(PM$_{2.5}$ emissions [tons day$^{-1}$])", "A", r"PM$_{2.5}$ Emissions")

violin_panel(axes[1], fire_match_df, "log_area",
             r"$\log_{10}(\mathrm{Burned\ Area\ (km^2)})$", "B", "Burned Area")

fig.suptitle("Fire Size and Detection Status", fontsize=15, fontweight="bold", y=1.02)

plt.tight_layout()
savefig(fig, "fig0_size_vs_detection_violin.png")
plt.show()
```


    
![png](output_39_0.png)
    


**Figure 5.** Total EPA fire events by type, restricted to June 2022 (the study period).


```python
counts_june = epa_june["type"].value_counts().reindex(FIRE_ORDER)
colors_june = [FIRE_COLORS[t] for t in FIRE_ORDER]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(counts_june.index, counts_june.values, color=colors_june, edgecolor="black")
add_bar_labels(ax, counts_june.values)
ax.set_ylim(0, counts_june.max() * 1.2)
ax.set_title("Number of Fire Events by Type in Georgia (June 2022)")
ax.set_xlabel("Fire Type")
ax.set_ylabel("Count")
sns.despine(ax=ax)

plt.tight_layout()
savefig(fig, "fig05_fire_counts_june2022.png")
plt.show()

(counts_june / counts_june.sum() * 100).round(1).rename("percent_of_june_fires")
```


    
![png](output_41_0.png)
    





    type
    Agricultural    35.7
    Prescribed      37.8
    Wildfire        26.5
    Name: percent_of_june_fires, dtype: float64




```python
other = epa.loc[
    epa.month != 6,
    "pm2.5"
]

june = epa.loc[
    epa.month == 6,
    "pm2.5"
]

u,p = mannwhitneyu(
    june,
    other,
    alternative="two-sided"
)

delta = cliffs_delta(
    june,
    other
)
print(
    f"MWU p={p:.3e}"
)

print(
    f"Cliff's delta={delta:.3f}"
)
```

    MWU p=1.334e-68
    Cliff's delta=-0.195



```python
# June representativeness relative to 2022

annual_pm = epa["pm2.5"].dropna()
june_pm = epa.loc[
    epa["month"] == 6,
    "pm2.5"
].dropna()

summary_compare = pd.DataFrame({
    "Annual": summarize_skewed(annual_pm),
    "June": summarize_skewed(june_pm)
})

display(summary_compare)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Annual</th>
      <th>June</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N</th>
      <td>79563.000000</td>
      <td>2771.000000</td>
    </tr>
    <tr>
      <th>Median</th>
      <td>0.150497</td>
      <td>0.071686</td>
    </tr>
    <tr>
      <th>Q1</th>
      <td>0.031567</td>
      <td>0.017868</td>
    </tr>
    <tr>
      <th>Q3</th>
      <td>0.806322</td>
      <td>0.388321</td>
    </tr>
    <tr>
      <th>P95</th>
      <td>5.135666</td>
      <td>2.214732</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>105554.733644</td>
      <td>2674.693335</td>
    </tr>
  </tbody>
</table>
</div>



```python
u,p = mannwhitneyu(
    june_pm,
    annual_pm,
    alternative="two-sided"
)

delta = cliffs_delta(
    june_pm,
    annual_pm
)

print(
    f"Mann-Whitney U p-value = {p:.3e}"
)

print(
    f"Cliff's Delta = {delta:.3f}"
)
```

    Mann-Whitney U p-value = 4.153e-64
    Cliff's Delta = -0.189


### 4.3 FINN Fire Activity During June 2022


```python
pm_plot = apply_ga_mask(pm_total.values, ga_mask)
pm_plot[pm_plot <= 1e-6] = np.nan
```


```python
annual_grid = np.array(pm_annual_total)

annual_grid = apply_ga_mask(
    annual_grid,
    ga_mask
)

annual_grid[
    annual_grid <= 1e-6
] = np.nan
```


```python
june_vals = pm_plot[np.isfinite(pm_plot)]
annual_vals = annual_grid[np.isfinite(annual_grid)]

all_vals = np.concatenate([june_vals, annual_vals])

common_vmin = np.nanpercentile(all_vals, 1)
common_vmax = np.nanpercentile(all_vals, 99)

norm = LogNorm(vmin=common_vmin, vmax=common_vmax)
```

**Figure 6.** Total FINN PM$_{2.5}$ emissions for Georgia, June 2022.


```python
fig, ax = basemap()

mesh = ax.pcolormesh(
    lon, lat, pm_plot, cmap="YlOrRd", norm=norm,
    transform=ccrs.PlateCarree(), shading="nearest", zorder=2
)

cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, extend="neither")
cbar.set_label(r"PM$_{2.5}$ emissions (tons)")

ax.set_title("FINN PM$_{2.5}$ Emissions (June 2022)", fontsize=13)

plt.tight_layout()
savefig(fig, "fig06a_finn_pm25_emissions.png")
plt.show()
```


    
![png](output_50_0.png)
    


***Figure 6B.***
Annual FINN PM2.5 Emissions (2022)


```python
fig, ax = basemap()

mesh = ax.pcolormesh(
    lon, lat, annual_grid, cmap="YlOrRd", norm=norm,
    transform=ccrs.PlateCarree(), shading="nearest", zorder=2
)

cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, extend="neither")
cbar.set_label(r"PM$_{2.5}$ emissions (tons)")

ax.set_title("FINN PM$_{2.5}$ Emissions (2022)", fontsize=13)

plt.tight_layout()
savefig(fig, "fig06b_finn_annual_pm25_emissions.png")
plt.show()
```


    
![png](output_52_0.png)
    



```python
monthly_finn = (
    finn_annual["pm_daily"]
    .groupby("time.month")
    .sum()
)
```

***Figure 7*** Monthly PM 2.5 Emissions in Georgia 2022


```python
plt.figure(figsize=(8,4))

plt.plot(
    monthly_finn["month"],
    monthly_finn,
    marker="o",
    linewidth=2
)

plt.axvline(
    6,
    ls="--",
    color="red",
    alpha=0.7
)

plt.title(
    "Monthly FINN PM$_{2.5}$ Emissions (2022)"
)

plt.ylabel("PM$_{2.5}$ (tons)")
plt.xlabel("Month")

plt.tight_layout()
savefig(fig, "fig07_finn_monthly_pm25_emissions.png")
plt.show()
```


    
![png](output_55_0.png)
    


### 4.4 Spatial Agreement Between EPA and FINN


```python
# --- grid edges & per-day FINN/EPA lookups for the spatial miss-rate maps ---
dlat = lat[1] - lat[0]
dlon = lon[1] - lon[0]
lat_edges = np.concatenate([[lat[0] - dlat / 2], lat + dlat / 2])
lon_edges = np.concatenate([[lon[0] - dlon / 2], lon + dlon / 2])

epa_dates = pd.to_datetime(gdf_june["date"]).dt.floor("D").values
epa_lat_pts = gdf_june.geometry.y.values
epa_lon_pts = gdf_june.geometry.x.values
epa_type_pts = gdf_june["type"].values

unique_dates, inverse_idx = np.unique(epa_dates, return_inverse=True)
date_groups = [np.where(inverse_idx == i)[0] for i in range(len(unique_dates))]

finn_times = pd.to_datetime(finn_data["time"].values).floor("D").values
time_lookup = np.array([np.argmin(np.abs(finn_times - d)) for d in unique_dates])

ga_mask_bool = ga_mask.astype(np.bool_)
nlat, nlon = len(lat), len(lon)


@njit(parallel=True)
def _apply_mask_fast(grid, mask):
    for i in prange(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not mask[i, j]:
                grid[i, j] = np.nan
    return grid


@njit
def _compute_miss_fast(epa_grid, finn_grid, threshold=1e-3):
    miss = np.zeros_like(epa_grid)
    for i in range(epa_grid.shape[0]):
        for j in range(epa_grid.shape[1]):
            if epa_grid[i, j] > 0 and finn_grid[i, j] <= threshold:
                miss[i, j] = 1
    return miss
```


```python
miss_ag = np.zeros((nlat, nlon))
miss_prescribed = np.zeros((nlat, nlon))
miss_wildfire = np.zeros((nlat, nlon))
miss_total = np.zeros((nlat, nlon))

for i in range(len(unique_dates)):
    inds = date_groups[i]
    if len(inds) == 0:
        continue

    lat_pts, lon_pts, types = epa_lat_pts[inds], epa_lon_pts[inds], epa_type_pts[inds]

    epa_grid, _, _ = np.histogram2d(lat_pts, lon_pts, bins=[lat_edges, lon_edges])
    epa_grid = _apply_mask_fast(epa_grid, ga_mask_bool)

    finn_grid = pm_tons.isel(time=time_lookup[i]).values.copy()
    finn_grid = _apply_mask_fast(finn_grid, ga_mask_bool)

    miss = _compute_miss_fast(epa_grid, finn_grid)

    for type_name, target in [
        ("Agricultural", miss_ag),
        ("Prescribed", miss_prescribed),
        ("Wildfire", miss_wildfire),
    ]:
        type_mask = types == type_name
        if np.any(type_mask):
            type_grid, _, _ = np.histogram2d(
                lat_pts[type_mask], lon_pts[type_mask], bins=[lat_edges, lon_edges]
            )
            target += (type_grid > 0) * miss

    miss_total += miss

for grid in (miss_ag, miss_prescribed, miss_wildfire, miss_total):
    grid[grid == 0] = np.nan

print("Total missed grid-days:", np.nansum(miss_total))
print("  Agricultural:", np.nansum(miss_ag))
print("  Prescribed  :", np.nansum(miss_prescribed))
print("  Wildfire    :", np.nansum(miss_wildfire))
```

    Total missed grid-days: 1623.0
      Agricultural: 848.0
      Prescribed  : 468.0
      Wildfire    : 343.0


***Figure 8*** Spatial distribution of missed EPA fires (grid-cell counts, summed over June), overall and split by fire type


```python
fig, axes = plt.subplots(
    2, 2, figsize=(10, 8),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

panels = [
    ("Total Missed Fires", miss_total),
    ("Agricultural", miss_ag),
    ("Prescribed", miss_prescribed),
    ("Wildfire", miss_wildfire),
]

for ax, (title, grid) in zip(axes.ravel(), panels):
    basemap(ax)
    mesh = ax.pcolormesh(
        lon, lat, grid, transform=ccrs.PlateCarree(),
        shading="nearest", cmap=cmap_trim, zorder=5
    )
    ax.set_title(title, fontsize=12)
    plt.colorbar(mesh, ax=ax, shrink=0.8, label="Missed fire-days")

fig.suptitle("Spatial distribution of missed EPA fires (June 2022)", fontsize=15, fontweight="bold", y=1.0)

plt.tight_layout()
savefig(fig, "fig08_miss_rate_by_type_maps.png")
plt.show()
```


    
![png](output_60_0.png)
    


#### Spatiotemporal Correlation


```python
daily_spatial_corr = []

for date in sorted(
    pd.to_datetime(
        gdf_june["date"]
    ).dt.normalize().unique()
):

    epa_day = get_daily_epa(
        gdf_june,
        date
    )

    epa_grid = epa_to_finn_grid(
        epa_day,
        lat,
        lon,
        "pm25"
    )

    epa_grid = apply_ga_mask(
        epa_grid,
        ga_mask
    )

    finn_day = (
        finn_data["pm_tons"]
        .sel(
            time=date,
            method="nearest"
        )
        .values
    )

    finn_day = apply_ga_mask(
        finn_day,
        ga_mask
    )

    r = spatial_correlation(
        epa_grid,
        finn_day
    )

    daily_spatial_corr.append({
        "date": date,
        "corr": r
    })

daily_spatial_corr = pd.DataFrame(
    daily_spatial_corr
)
```


```python
daily_spatial_corr["corr"].describe(
    percentiles=[.25,.5,.75,.95]
)
```




    count    15.000000
    mean      0.220475
    std       0.428437
    min      -0.418055
    25%      -0.129785
    50%       0.230402
    75%       0.610104
    95%       0.750954
    max       0.825618
    Name: corr, dtype: float64




```python
median_corr = daily_spatial_corr["corr"].median()
mean_corr = daily_spatial_corr["corr"].mean()

print(f"Mean correlation: {mean_corr:.3f}")
print(f"Median correlation: {median_corr:.3f}")
```

    Mean correlation: 0.220
    Median correlation: 0.230



```python
positive_days = (
    daily_spatial_corr["corr"] > 0
).mean() * 100

print(
    f"Days with positive spatial correlation: "
    f"{positive_days:.1f}%"
)
```

    Days with positive spatial correlation: 33.3%


***Figure 10.*** Daily EPA–FINN Spatiotemporal Correlation


```python
fig, ax = plt.subplots(
    figsize=(9,4)
)

ax.scatter(
    daily_spatial_corr["date"],
    daily_spatial_corr["corr"],
    s=60,
    color="black"
)

ax.axhline(
    daily_spatial_corr["corr"].median(),
    color="red",
    linestyle="--"
)
fig.suptitle("Daily Spatial Correlation", fontsize=15, fontweight="bold", y=1.0)

plt.tight_layout()
savefig(fig, "fig10_daily_spatial_corr.png")
plt.show()
```


    
![png](output_67_0.png)
    



```python
all_epa = []
all_finn = []

for date in sorted(
    pd.to_datetime(
        gdf_june["date"]
    ).dt.normalize().unique()
):

    epa_day = get_daily_epa(
        gdf_june,
        date
    )

    epa_grid = epa_to_finn_grid(
        epa_day,
        lat,
        lon,
        "pm25"
    )

    epa_grid = apply_ga_mask(
        epa_grid,
        ga_mask
    )

    finn_day = (
        finn_data["pm_tons"]
        .sel(
            time=date,
            method="nearest"
        )
        .values
    )

    finn_day = apply_ga_mask(
        finn_day,
        ga_mask
    )

    valid = (
        np.isfinite(epa_grid)
        & np.isfinite(finn_day)
    )

    all_epa.extend(
        epa_grid[valid].ravel()
    )

    all_finn.extend(
        finn_day[valid].ravel()
    )

all_epa = np.array(all_epa)
all_finn = np.array(all_finn)

bias = np.mean(
    all_finn - all_epa
)

rmse = np.sqrt(
    np.mean(
        (all_finn - all_epa) ** 2
    )
)

print("Bias:", bias)
print("RMSE:", rmse)
```

    Bias: 0.02490922210615998
    RMSE: 3.8767006879992056


### 4.5 Detection Performance

#### Detection performance: confusion matrix


```python
overall_metrics = run_daily_spatiotemporal_analysis(gdf_june, finn_data, lat, lon)

rows = []
for fire_type in FIRE_ORDER:
    subset = gdf_june[gdf_june["type"] == fire_type]
    m = run_daily_spatiotemporal_analysis(subset, finn_data, lat, lon)
    rows.append({
        "Fire Type": fire_type,
        "TP": m["TP"], "FN": m["FN"], "FP": m["FP"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "Miss Rate": m["FN"] / (m["TP"] + m["FN"] + 1e-6),
    })

confusion_summary = pd.DataFrame(rows).set_index("Fire Type")
confusion_summary[["Precision", "Recall", "Miss Rate"]] = (
    confusion_summary[["Precision", "Recall", "Miss Rate"]].round(3)
)
confusion_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>FN</th>
      <th>FP</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Miss Rate</th>
    </tr>
    <tr>
      <th>Fire Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Agricultural</th>
      <td>67</td>
      <td>848</td>
      <td>452</td>
      <td>0.129</td>
      <td>0.073</td>
      <td>0.927</td>
    </tr>
    <tr>
      <th>Prescribed</th>
      <td>57</td>
      <td>468</td>
      <td>462</td>
      <td>0.110</td>
      <td>0.109</td>
      <td>0.891</td>
    </tr>
    <tr>
      <th>Wildfire</th>
      <td>27</td>
      <td>343</td>
      <td>492</td>
      <td>0.052</td>
      <td>0.073</td>
      <td>0.927</td>
    </tr>
  </tbody>
</table>
</div>



#### Detection by fire types


```python
type_detection = (
    fire_match_df
    .groupby("type")["matched"]
    .agg(
        matches="sum",
        count="count"
    )
    .reindex(FIRE_ORDER)
)

type_detection["rate"] = (
    type_detection["matches"]
    /
    type_detection["count"]
)

ci_low = []
ci_high = []

for k, n in zip(
    type_detection["matches"],
    type_detection["count"]
):

    low, high = proportion_confint(
        count=k,
        nobs=n,
        alpha=0.05,
        method="wilson"
    )

    ci_low.append(low)
    ci_high.append(high)

type_detection["ci_low"] = ci_low
type_detection["ci_high"] = ci_high

display(type_detection)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>matches</th>
      <th>count</th>
      <th>rate</th>
      <th>ci_low</th>
      <th>ci_high</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Agricultural</th>
      <td>78</td>
      <td>989</td>
      <td>0.078868</td>
      <td>0.063652</td>
      <td>0.097342</td>
    </tr>
    <tr>
      <th>Prescribed</th>
      <td>131</td>
      <td>1048</td>
      <td>0.125000</td>
      <td>0.106336</td>
      <td>0.146403</td>
    </tr>
    <tr>
      <th>Wildfire</th>
      <td>51</td>
      <td>734</td>
      <td>0.069482</td>
      <td>0.053240</td>
      <td>0.090207</td>
    </tr>
  </tbody>
</table>
</div>


*** Figure 9 *** Fraction of EPA fires matched to FINN emissions by fire type.
Bars show the proportion of EPA fire records that were associated with at
least one FINN PM2.5-emitting grid cell during June 2022. Error bars
indicate 95% confidence intervals.


```python
fig, ax = plt.subplots(
    figsize=(6,4.5)
)

bars = ax.bar(
    type_detection.index,
    type_detection["rate"],
    color=[
        FIRE_COLORS[t]
        for t in type_detection.index
    ],
    edgecolor="black",
    linewidth=1
)

lower = (
    type_detection["rate"]
    -
    type_detection["ci_low"]
)

upper = (
    type_detection["ci_high"]
    -
    type_detection["rate"]
)

ax.errorbar(
    np.arange(len(type_detection)),
    type_detection["rate"],
    yerr=[lower, upper],
    fmt="none",
    color="black",
    capsize=4
)

for i, (
    rate,
    n,
    upper_ci
) in enumerate(zip(
    type_detection["rate"],
    type_detection["count"],
    type_detection["ci_high"]
)):

    ax.text(
        i,
        upper_ci + 0.005,
        f"{rate:.1%}",
        ha="center",
        fontweight="bold"
    )

    ax.text(
        i,
        0.003,
        f"n={n}",
        ha="center",
        fontsize=9,
        color="white"
    )

ax.set_ylabel(
    "Detection Probability"
)

ax.set_title(
    "EPA–FINN Match Rate by Fire Type"
)

ax.yaxis.set_major_formatter(
    PercentFormatter(1)
)

ax.set_ylim(
    0,
    type_detection["ci_high"].max()
    * 1.4
)

sns.despine()

plt.tight_layout()

savefig(
    fig,
    "fig09_detection_wilson.png"
)

plt.show()
```


    
![png](output_75_0.png)
    


#### sensitivity test


```python
neighbor_results = []

for r in [0, 1]:

    m = run_daily_spatiotemporal_analysis_neighbor(
        gdf_june,
        finn_data,
        lat,
        lon,
        radius=r
    )

    neighbor_results.append({
        "Radius": r,
        "TP": m["TP"],
        "FN": m["FN"],
        "FP": m["FP"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"]
    })

neighbor_results = pd.DataFrame(
    neighbor_results
)

neighbor_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Radius</th>
      <th>TP</th>
      <th>FN</th>
      <th>FP</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>139</td>
      <td>1623</td>
      <td>380</td>
      <td>0.267823</td>
      <td>0.078888</td>
      <td>0.121876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>354</td>
      <td>1408</td>
      <td>253</td>
      <td>0.583196</td>
      <td>0.200908</td>
      <td>0.298860</td>
    </tr>
  </tbody>
</table>
</div>



**Table 1.** Daily spatiotemporal detection metrics (grid-cell TP/FN/FP, using `run_daily_spatiotemporal_analysis`) by fire type, with a combined total row and sensitivity test

#### Spatial Offset Between EPA and FINN


```python
from scipy.spatial import cKDTree

distance_rows = []

for date in sorted(
    pd.to_datetime(gdf_june["date"])
    .dt.normalize()
    .unique()
):

    epa_day = gdf_june[
        pd.to_datetime(gdf_june["date"])
        .dt.normalize() == date
    ]

    finn_day = (
        finn_data["pm_tons"]
        .sel(time=date, method="nearest")
        .values
    )

    active = finn_day > 0

    if active.sum() == 0:
        continue

    finn_pts = np.column_stack([
        np.repeat(lat, len(lon))[active.ravel()],
        np.tile(lon, len(lat))[active.ravel()]
    ])

    tree = cKDTree(finn_pts)

    epa_pts = np.column_stack([
        epa_day.geometry.y.values,
        epa_day.geometry.x.values
    ])

    dist_deg, _ = tree.query(epa_pts)

    distance_rows.extend(
        list(dist_deg * 111)
    )

distance_df = pd.DataFrame({
    "distance_km": distance_rows
})

distance_df.describe(
    percentiles=[0.25,0.5,0.75,0.95]
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distance_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2771.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49.387353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>39.717469</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.149990</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.906455</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.943795</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65.415742</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>128.810954</td>
    </tr>
    <tr>
      <th>max</th>
      <td>348.266649</td>
    </tr>
  </tbody>
</table>
</div>



### 4.6 Influence of Fire Size on Detection

***Figure 10*** EPA–FINN match rates by fire-size quartile. Bars show the fraction of EPA fire records matched to FINN emissions, grouped by quartiles of reported PM₂.₅ emissions (A) and burned area (B). Error bars denote 95% Wilson confidence intervals for binomial proportions. Sample sizes are shown within bars. Match rates increase monotonically with both emissions and burned area, indicating that larger fires are more likely to be represented in the FINN inventory.


```python
fire_match_df["pm_bin"] = pd.qcut(
    fire_match_df["pm2.5"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop"
)

fire_match_df["area_bin"] = pd.qcut(
    fire_match_df["area_km2"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop"
)

def match_rate_summary(df, group_col):

    summary = (
        df.groupby(group_col, observed=True)["matched"]
        .agg(matches="sum", count="count")
        .reset_index()
    )

    summary["rate"] = summary["matches"] / summary["count"]

    ci_low = []
    ci_high = []

    for k, n in zip(summary["matches"], summary["count"]):

        low, high = proportion_confint(
            count=k,
            nobs=n,
            alpha=0.05,
            method="wilson"
        )

        ci_low.append(low)
        ci_high.append(high)

    summary["ci_low"] = ci_low
    summary["ci_high"] = ci_high

    return summary


pm_summary = match_rate_summary(fire_match_df, "pm_bin")
area_summary = match_rate_summary(fire_match_df, "area_bin")

pm_edges = fire_match_df["pm2.5"].quantile([0, .25, .5, .75, 1]).values
area_edges = fire_match_df["area_km2"].quantile([0, .25, .5, .75, 1]).values

pm_xticklabels = ["Q1", "Q2", "Q3", "Q4"]
area_xticklabels = ["Q1", "Q2", "Q3", "Q4"]

def plot_match_rate(ax, summary, xticklabels, xlabel, color):

    x = np.arange(len(summary))

    bars = ax.bar(
        x,
        summary["rate"],
        color=color,
        edgecolor="black",
        linewidth=1
    )

    lower = summary["rate"] - summary["ci_low"]
    upper = summary["ci_high"] - summary["rate"]

    ax.errorbar(
        x,
        summary["rate"],
        yerr=[lower, upper],
        fmt="none",
        color="black",
        capsize=4,
        lw=1.2
    )

    for xx, rate, hi, n in zip(
        x,
        summary["rate"],
        summary["ci_high"],
        summary["count"]
    ):

        ax.text(
            xx,
            hi + 0.015,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            xx,
            0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Match Rate")

    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 0.40)

    sns.despine(ax=ax)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

plot_match_rate(
    axes[0],
    pm_summary,
    pm_xticklabels,
    r"PM$_{2.5}$ Emissions Quartile (tons day$^{-1}$)",
    "#6baed6"
)
axes[0].set_title("A. By PM$_{2.5}$ Emissions", loc="left")

plot_match_rate(
    axes[1],
    area_summary,
    area_xticklabels,
    r"Burned Area Quartile (km$^2$)",
    "#74c476"
)
axes[1].set_title("B. By Burned Area", loc="left")

fig.suptitle(
    "EPA–FINN Match Rate Across Fire-Size Quartiles",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

savefig(fig, "fig10_match_rate_by_size_quartiles.png")
plt.show()
```


    
![png](output_83_0.png)
    



```python
fire_match_df["log_pm"] = np.log10(fire_match_df["pm2.5"] + 1e-6)
fire_match_df["log_area"] = np.log10(fire_match_df["area_km2"] + 1e-6)
```

Logistic regression: PM2.5.


```python
pm_model_log = smf.logit(
    "matched ~ log_pm",
    data=fire_match_df
).fit()

print(pm_model_log.summary())
```

    Optimization terminated successfully.
             Current function value: 0.302317
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                matched   No. Observations:                 2771
    Model:                          Logit   Df Residuals:                     2769
    Method:                           MLE   Df Model:                            1
    Date:                Tue, 14 Jul 2026   Pseudo R-squ.:                 0.02888
    Time:                        14:59:23   Log-Likelihood:                -837.72
    converged:                       True   LL-Null:                       -862.64
    Covariance Type:            nonrobust   LLR p-value:                 1.677e-12
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -1.8010      0.086    -20.956      0.000      -1.969      -1.633
    log_pm         0.4859      0.070      6.928      0.000       0.348       0.623
    ==============================================================================


Logistic regression: area.


```python
area_model_log = smf.logit(
    "matched ~ log_area",
    data=fire_match_df
).fit()

print(area_model_log.summary())
```

    Optimization terminated successfully.
             Current function value: 0.297818
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                matched   No. Observations:                 2771
    Model:                          Logit   Df Residuals:                     2769
    Method:                           MLE   Df Model:                            1
    Date:                Tue, 14 Jul 2026   Pseudo R-squ.:                 0.04333
    Time:                        14:59:23   Log-Likelihood:                -825.25
    converged:                       True   LL-Null:                       -862.64
    Covariance Type:            nonrobust   LLR p-value:                 5.315e-18
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -1.2464      0.125     -9.952      0.000      -1.492      -1.001
    log_area       0.6444      0.076      8.425      0.000       0.494       0.794
    ==============================================================================



```python
logistic_summary = pd.DataFrame({

    "Predictor":[
        "PM2.5",
        "Burned Area"
    ],

    "Odds Ratio":[
        np.exp(pm_model_log.params.iloc[1]),
        np.exp(area_model_log.params.iloc[1])
    ],

    "Pseudo R²":[
        pm_model_log.prsquared,
        area_model_log.prsquared
    ],

    "p-value":[
        pm_model_log.pvalues.iloc[1],
        area_model_log.pvalues.iloc[1]
    ]
})

display(
    logistic_summary.round(3)
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictor</th>
      <th>Odds Ratio</th>
      <th>Pseudo R²</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PM2.5</td>
      <td>1.626</td>
      <td>0.029</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Burned Area</td>
      <td>1.905</td>
      <td>0.043</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
fire_match_df["log_pm"] = np.log10(
    fire_match_df["pm2.5"] + 1e-6
)

fire_match_df["log_area"] = np.log10(
    fire_match_df["area_km2"] + 1e-6
)

multi_model = smf.logit(
    "matched ~ log_pm + log_area + C(type)",
    data=fire_match_df
).fit()

print(multi_model.summary())

multi_results = pd.DataFrame({
    "Variable": multi_model.params.index,
    "Odds Ratio": np.exp(multi_model.params),
    "p-value": multi_model.pvalues
})

display(
    multi_results.round(3)
)

print(
    "Pseudo R²:",
    round(multi_model.prsquared,3)
)
```

    Optimization terminated successfully.
             Current function value: 0.294871
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                matched   No. Observations:                 2771
    Model:                          Logit   Df Residuals:                     2766
    Method:                           MLE   Df Model:                            4
    Date:                Tue, 14 Jul 2026   Pseudo R-squ.:                 0.05280
    Time:                        14:59:24   Log-Likelihood:                -817.09
    converged:                       True   LL-Null:                       -862.64
    Covariance Type:            nonrobust   LLR p-value:                 7.704e-19
    =========================================================================================
                                coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                -1.5335      0.161     -9.498      0.000      -1.850      -1.217
    C(type)[T.Prescribed]     0.6070      0.153      3.967      0.000       0.307       0.907
    C(type)[T.Wildfire]       0.3585      0.200      1.795      0.073      -0.033       0.750
    log_pm                   -0.0453      0.117     -0.387      0.699      -0.275       0.184
    log_area                  0.7088      0.136      5.199      0.000       0.442       0.976
    =========================================================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Odds Ratio</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>Intercept</td>
      <td>0.216</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>C(type)[T.Prescribed]</th>
      <td>C(type)[T.Prescribed]</td>
      <td>1.835</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>C(type)[T.Wildfire]</th>
      <td>C(type)[T.Wildfire]</td>
      <td>1.431</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>log_pm</th>
      <td>log_pm</td>
      <td>0.956</td>
      <td>0.699</td>
    </tr>
    <tr>
      <th>log_area</th>
      <td>log_area</td>
      <td>2.032</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>


    Pseudo R²: 0.053


### 4.7 Emissions and Burned Area Captured by FINN

**Table 2.** Total, detected, and missed PM$_{2.5}$ emissions and burned area by fire type, with a combined total row — and the EPA vs. FINN total-emissions comparison referenced in the discussion.


```python
def emissions_by_type(df):
    total = df.groupby("type").agg(
        total_pm25=("pm2.5", "sum"),
        total_area_km2=("area_km2", "sum"),
    )
    detected = df[df["matched"] == 1].groupby("type").agg(
        detected_pm25=("pm2.5", "sum"), detected_area_km2=("area_km2", "sum")
    )
    missed = df[df["matched"] == 0].groupby("type").agg(
        missed_pm25=("pm2.5", "sum"), missed_area_km2=("area_km2", "sum")
    )
    out = total.join(detected).join(missed).fillna(0)
    out["pct_missed_pm25"] = out["missed_pm25"] / out["total_pm25"] * 100
    out["pct_missed_area"] = out["missed_area_km2"] / out["total_area_km2"] * 100
    return out


emissions_table = emissions_by_type(fire_match_df).reindex(FIRE_ORDER)

total_row = pd.DataFrame({
    "total_pm25": [emissions_table["total_pm25"].sum()],
    "total_area_km2": [emissions_table["total_area_km2"].sum()],
    "detected_pm25": [emissions_table["detected_pm25"].sum()],
    "detected_area_km2": [emissions_table["detected_area_km2"].sum()],
    "missed_pm25": [emissions_table["missed_pm25"].sum()],
    "missed_area_km2": [emissions_table["missed_area_km2"].sum()],
}, index=["Total (Combined)"])
total_row["pct_missed_pm25"] = total_row["missed_pm25"] / total_row["total_pm25"] * 100
total_row["pct_missed_area"] = total_row["missed_area_km2"] / total_row["total_area_km2"] * 100

emissions_table = pd.concat([emissions_table, total_row])
emissions_table.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_pm25</th>
      <th>total_area_km2</th>
      <th>detected_pm25</th>
      <th>detected_area_km2</th>
      <th>missed_pm25</th>
      <th>missed_area_km2</th>
      <th>pct_missed_pm25</th>
      <th>pct_missed_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Agricultural</th>
      <td>369.00</td>
      <td>112.13</td>
      <td>48.09</td>
      <td>15.60</td>
      <td>320.91</td>
      <td>96.53</td>
      <td>86.97</td>
      <td>86.09</td>
    </tr>
    <tr>
      <th>Prescribed</th>
      <td>1550.95</td>
      <td>167.18</td>
      <td>766.80</td>
      <td>71.92</td>
      <td>784.15</td>
      <td>95.26</td>
      <td>50.56</td>
      <td>56.98</td>
    </tr>
    <tr>
      <th>Wildfire</th>
      <td>754.74</td>
      <td>39.74</td>
      <td>50.40</td>
      <td>3.36</td>
      <td>704.35</td>
      <td>36.38</td>
      <td>93.32</td>
      <td>91.54</td>
    </tr>
    <tr>
      <th>Total (Combined)</th>
      <td>2674.69</td>
      <td>319.04</td>
      <td>865.28</td>
      <td>90.88</td>
      <td>1809.41</td>
      <td>228.16</td>
      <td>67.65</td>
      <td>71.51</td>
    </tr>
  </tbody>
</table>
</div>



#### PM 2.5 Capture


```python
# fire classification
annual_fire_match_df = classify_all_days(
    gdf_annual,
    finn_annual,
    lat,
    lon
)

annual_fire_match_df["area_km2"] = (
    annual_fire_match_df["area"]
    *
    ACRES_TO_KM2
)

annual_fire_match_df["Detection Status"] = (
    annual_fire_match_df["matched"]
    .map({
        1:"Detected",
        0:"Missed"
    })
)

print(
    f"Annual fires: {len(annual_fire_match_df):,}"
)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[53], line 2
          1 # fire classification
    ----> 2 annual_fire_match_df = classify_all_days(
          3     gdf_annual,
          4     finn_annual,
          5     lat,
          6     lon
          7 )
          9 annual_fire_match_df["area_km2"] = (
         10     annual_fire_match_df["area"]
         11     *
         12     ACRES_TO_KM2
         13 )
         15 annual_fire_match_df["Detection Status"] = (
         16     annual_fire_match_df["matched"]
         17     .map({
       (...)     20     })
         21 )


    Cell In[14], line 9, in classify_all_days(gdf, finn, lat, lon)
          7         continue
          8     finn_day = finn["pm_tons"].sel(time=date, method="nearest").values
    ----> 9     results.append(classify_missed_fires(epa_day, finn_day, lat, lon))
         10 return pd.concat(results, ignore_index=True)


    File ~/src/spatial_utils.py:86, in classify_missed_fires(epa_gdf, finn_grid, lat, lon, threshold)
         79 def classify_missed_fires(epa_gdf, finn_grid, lat, lon, threshold=0):
         80     """
         81     Classify EPA fires as matched (detected) or missed by FINN.
         82 
         83     threshold : minimum FINN value to count as detection
         84     """
    ---> 86     tree, lat_mesh = build_kdtree(lat, lon)
         88     coords = np.column_stack([
         89         epa_gdf.geometry.y,
         90         epa_gdf.geometry.x
         91     ])
         93     _, idx = tree.query(coords)


    File ~/src/spatial_utils.py:22, in build_kdtree(lat, lon)
         20 lon_mesh, lat_mesh = np.meshgrid(lon, lat)
         21 grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    ---> 22 tree = cKDTree(grid_points)
         23 return tree, lat_mesh


    KeyboardInterrupt: 



```python
pm_capture = (
    annual_fire_match_df
    .groupby("type")
    .apply(
        lambda x: pd.Series({

            "Total PM25":
            x["pm2.5"].sum(),

            "Detected PM25":
            x.loc[
                x["matched"] == 1,
                "pm2.5"
            ].sum()

        })
    )
)
```


```python
pm_capture[
    "Capture Rate"
] = (
    pm_capture["Detected PM25"]
    /
    pm_capture["Total PM25"]
)

pm_capture = (
    pm_capture
    .reindex(FIRE_ORDER)
)

pm_capture
```

*** Figure 11. *** PM2.5 capture by fire type.


```python
fig, ax = plt.subplots(
    figsize=(6,4)
)

bars = ax.bar(
    pm_capture.index,
    pm_capture["Capture Rate"],
    color=[
        FIRE_COLORS[t]
        for t
        in pm_capture.index
    ],
    edgecolor="black"
)

for bar, value in zip(
    bars,
    pm_capture["Capture Rate"]
):

    ax.text(
        bar.get_x()
        +
        bar.get_width()/2,
        value+0.01,
        f"{value:.1%}",
        ha="center",
        fontweight="bold"
    )

ax.set_ylabel(
    "PM₂.₅ Capture"
)

ax.yaxis.set_major_formatter(
    PercentFormatter(1)
)

ax.set_title(
    "PM₂.₅ Capture by Fire Type"
)

plt.tight_layout()
savefig(fig, "fig00_pm_capture_type.png")
plt.show()
```

#### Burned Area Capture


```python
area_capture = (
    annual_fire_match_df
    .groupby("type")
    .apply(
        lambda x: pd.Series({

            "Total Area":
            x["area_km2"].sum(),

            "Detected Area":
            x.loc[
                x["matched"] == 1,
                "area_km2"
            ].sum()

        })
    )
)
```


```python
area_capture[
    "Capture Rate"
] = (
    area_capture["Detected Area"]
    /
    area_capture["Total Area"]
)

area_capture = (
    area_capture
    .reindex(FIRE_ORDER)
)

area_capture
```

*** Figure 12 *** Burned-area capture by fire type.


```python
fig, ax = plt.subplots(
    figsize=(6,4)
)

bars = ax.bar(
    area_capture.index,
    area_capture["Capture Rate"],
    color=[
        FIRE_COLORS[t]
        for t
        in area_capture.index
    ],
    edgecolor="black"
)

for bar, value in zip(
    bars,
    area_capture["Capture Rate"]
):

    ax.text(
        bar.get_x()
        +
        bar.get_width()/2,
        value+0.01,
        f"{value:.1%}",
        ha="center",
        fontweight="bold"
    )

ax.set_ylabel(
    "Area Capture"
)

ax.yaxis.set_major_formatter(
    PercentFormatter(1)
)

ax.set_title(
    "Burned Area Capture by Fire Type"
)

plt.tight_layout()
savefig(fig, "fig000_area_capture_type.png")
plt.show()
```

### 4.8 June Versus Annual Detection Skill


```python
monthly_skill = []

for month in range(1,13):

    gdf_month = gdf_annual[
        gdf_annual["month"] == month
    ]

    m = run_daily_spatiotemporal_analysis(
        gdf_month,
        finn_annual,
        lat,
        lon
    )

    precision = m["precision"]
    recall = m["recall"]

    f1 = (
        2*precision*recall
        /
        (precision+recall+1e-6)
    )

    monthly_skill.append({
        "month": month,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

monthly_skill = pd.DataFrame(
    monthly_skill
)
```

***Figure 11.*** "Monthly FINN Detection Performance


```python
fig, ax = plt.subplots(
    figsize=(9,5)
)

ax.plot(
    monthly_skill["month"],
    monthly_skill["precision"],
    marker="o",
    label="Precision"
)

ax.plot(
    monthly_skill["month"],
    monthly_skill["recall"],
    marker="o",
    label="Recall"
)

ax.plot(
    monthly_skill["month"],
    monthly_skill["f1"],
    marker="o",
    linewidth=3,
    label="F1"
)

ax.axvline(
    6,
    color="red",
    ls="--"
)

ax.set_xticks(
    range(1,13)
)

ax.set_xticklabels(
    MONTHS_LABELS
)

ax.set_ylabel(
    "Skill"
)

ax.set_title(
    "Monthly FINN Detection Performance"
)

ax.legend()

plt.tight_layout()
savefig(fig, "fig11_monthly_detection_performance.png")
plt.show()
```


```python
tp = overall_metrics["TP"]

fp = overall_metrics["FP"]

fn = overall_metrics["FN"]

june_precision = tp/(tp+fp+1e-6)

june_recall = tp/(tp+fn+1e-6)

june_f1 = (
    2*june_precision*june_recall
    /
    (
        june_precision
        +
        june_recall
        +
        1e-6
    )
)

june_jaccard = (
    tp
    /
    (
        tp+fp+fn+1e-6
    )
)

june_dice = (
    2*tp
    /
    (
        2*tp+fp+fn+1e-6
    )
)
```


```python
# annual confusion matrix
annual_metrics = run_daily_spatiotemporal_analysis(
    gdf_annual,
    finn_annual,
    lat,
    lon
)
```


```python
tp = annual_metrics["TP"]

fp = annual_metrics["FP"]

fn = annual_metrics["FN"]

precision = tp/(tp+fp+1e-6)

recall = tp/(tp+fn+1e-6)

f1 = (
    2*precision*recall
    /
    (precision+recall+1e-6)
)

jaccard = (
    tp
    /
    (tp+fp+fn+1e-6)
)

dice = (
    2*tp
    /
    (
        2*tp+fp+fn+1e-6
    )
)

annual_detection_metrics = pd.Series({

    "Precision":precision,
    "Recall":recall,
    "F1":f1,
    "Jaccard":jaccard,
    "Dice":dice

})

display(
    annual_detection_metrics
)
```


```python
summary_table = pd.DataFrame({

    "Metric":[

        "Precision",
        "Recall",
        "F1",
        "Jaccard",
        "Dice"

    ],

    "June":[

        june_precision,
        june_recall,
        june_f1,
        june_jaccard,
        june_dice

    ],

    "Annual":[

        annual_detection_metrics["Precision"],
        annual_detection_metrics["Recall"],
        annual_detection_metrics["F1"],
        annual_detection_metrics["Jaccard"],
        annual_detection_metrics["Dice"]

    ]
})

summary_table
```

## Unknown


```python
pm_pred = pd.DataFrame({

    "log_pm": np.linspace(
        fire_match_df["log_pm"].min(),
        fire_match_df["log_pm"].max(),
        200
    )

})

pm_pred["prob"] = pm_model_log.predict(
    pm_pred
)

pm_pred.head()
```


```python
fig, ax = plt.subplots(
    figsize=(6,4.5)
)

ax.scatter(
    fire_match_df["log_pm"],
    fire_match_df["matched"],
    alpha=0.08,
    color="gray",
    s=15,
    label="EPA fires"
)

ax.plot(
    pm_pred["log_pm"],
    pm_pred["prob"],
    color="#2171b5",
    linewidth=3,
    label="Logistic fit"
)

ax.set_xlabel(
    r"log$_{10}$(PM$_{2.5}$ + 1e-6)"
)

ax.set_ylabel(
    "Detection Probability"
)

ax.set_ylim(
    0,
    1
)

ax.set_title(
    "Detection Probability vs PM$_{2.5}$ Emissions"
)

ax.legend()

sns.despine()

plt.tight_layout()

savefig(
    fig,
    "fig15a_logistic_pm25_detection.png"
)

plt.show()
```

For each 10-fold increase in PM2.5 emissions, 
the odds of FINN detection increase by1.63..


```python
area_pred = pd.DataFrame({

    "log_area": np.linspace(
        fire_match_df["log_area"].min(),
        fire_match_df["log_area"].max(),
        200
    )

})

area_pred["prob"] = area_model_log.predict(
    area_pred
)

area_pred.head()
```


```python
fig, ax = plt.subplots(
    figsize=(6,4.5)
)

ax.scatter(
    fire_match_df["log_area"],
    fire_match_df["matched"],
    alpha=0.08,
    color="gray",
    s=15,
    label="EPA fires"
)

ax.plot(
    area_pred["log_area"],
    area_pred["prob"],
    color="#31a354",
    linewidth=3,
    label="Logistic fit"
)

ax.set_xlabel(
    r"log$_{10}$(Burned Area km$^2$)"
)

ax.set_ylabel(
    "Detection Probability"
)

ax.set_ylim(
    0,
    1
)

ax.set_title(
    "Detection Probability vs Burned Area"
)

ax.legend()

sns.despine()

plt.tight_layout()

savefig(
    fig,
    "fig15b_logistic_area_detection.png"
)

plt.show()
```


```python
fig, ax = plt.subplots(
    figsize=(6,4.5)
)

ax.plot(
    pm_pred["log_pm"],
    pm_pred["prob"],
    color="#2171b5",
    linewidth=3,
    label="Logistic fit"
)

quartile_x = []

for q in ["Q1","Q2","Q3","Q4"]:

    x = np.median(
        fire_match_df.loc[
            fire_match_df["pm_bin"] == q,
            "log_pm"
        ]
    )

    quartile_x.append(x)

ax.errorbar(
    quartile_x,
    pm_summary["rate"],
    yerr=[
        pm_summary["rate"] - pm_summary["ci_low"],
        pm_summary["ci_high"] - pm_summary["rate"]
    ],
    fmt="o",
    color="black",
    markersize=7,
    capsize=4,
    label="Observed quartiles"
)

ax.set_xlabel(
    r"log$_{10}$(PM$_{2.5}$)"
)

ax.set_ylabel(
    "Detection Probability"
)

ax.set_ylim(0,1)

ax.legend()

sns.despine()

plt.tight_layout()

plt.show()
```

For each 10-fold increase in PM2.5 emissions, the odds of FINN detection increase by 1.9.


```python
# --- EPA vs. FINN total PM2.5, Georgia, June 2022 ---
ga_pm_finn = apply_ga_mask(pm_tons.values, ga_mask)
finn_total_pm25 = np.nansum(ga_pm_finn)
epa_total_pm25 = gdf_june["pm2.5"].sum()

bias = finn_total_pm25 - epa_total_pm25
ratio = finn_total_pm25 / (epa_total_pm25 + 1e-6)

print(f"EPA total PM2.5 (June, GA)  : {epa_total_pm25:,.1f} tons")
print(f"FINN total PM2.5 (June, GA) : {finn_total_pm25:,.1f} tons")
print(f"Difference (FINN - EPA)     : {bias:,.1f} tons")
print(f"Ratio (FINN / EPA)          : {ratio:.2f}")
```


```python
june_grid = np.array(
    finn_annual["pm_tons"]
    .sel(time=finn_annual["time"].dt.month == 6)
    .sum(dim="time")
)

june_grid = apply_ga_mask(
    june_grid,
    ga_mask
)

# june_total_pm25 = pm_total.sum().values

# annual_total_pm25 = finn_annual["pm_total"].sum().values
june_total_pm25 = np.nansum(
    apply_ga_mask(
        pm_total.values,
        ga_mask
    )
)

annual_total_pm25 = np.nansum(
    apply_ga_mask(
        finn_annual["pm_total"].values,
        ga_mask
    )
)

print(f"June total: {june_total_pm25:,.0f} tons")
print(f"Annual total: {annual_total_pm25:,.0f} tons")

print(
    f"June contribution = "
    f"{100*june_total_pm25/annual_total_pm25:.1f}%"
)
```


```python
fig, ax = plt.subplots(
    figsize=(7,5)
)

plot_df = pd.concat([

    pd.DataFrame({
        "PM25":annual_pm,
        "Group":"Annual"
    }),

    pd.DataFrame({
        "PM25":june_pm,
        "Group":"June"
    })

])

sns.violinplot(
    data=plot_df,
    x="Group",
    y="PM25",
    cut=0
)

ax.set_yscale("log")

ax.set_title(
    "June vs Annual PM2.5 Distributions"
)

plt.tight_layout()
```


```python
tp = overall_metrics["TP"]
fp = overall_metrics["FP"]
fn = overall_metrics["FN"]

precision = tp/(tp+fp)

recall = tp/(tp+fn)

f1 = (
    2*precision*recall
    /
    (precision+recall)
)

jaccard = (
    tp
    /
    (tp+fp+fn)
)

dice = (
    2*tp
    /
    (2*tp+fp+fn)
)

pd.Series({
    "Precision":precision,
    "Recall":recall,
    "F1":f1,
    "Jaccard":jaccard,
    "Dice":dice
})
```
