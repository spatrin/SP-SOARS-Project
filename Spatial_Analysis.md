```python
import sys
import os

project_root = "/glade/u/home/spatrin"

FIG_DIR = "/glade/u/home/spatrin/figures"
os.makedirs(FIG_DIR, exist_ok=True)

src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

print("Project root:", project_root)
print("Using src path:", src_path)
```

    Project root: /glade/u/home/spatrin
    Using src path: /glade/u/home/spatrin/src



```python
import os
print(os.listdir(src_path))
```

    ['data_loading.py', 'pipeline.py', '.ipynb_checkpoints', 'spatial_utils.py', 'utils.py', 'aggregation.py', 'conversion.py', 'geo_utils.py', '__pycache__']



```python
from spatial_utils import (
    epa_to_finn_grid,
    basemap, 
    classify_missed_fires,
    get_daily_epa,
    plot_finn_epa_overlay)
from geo_utils import *
from pipeline import load_finn_june_processed
from data_loading import load_epa_data, filter_june

import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
from numba import njit, prange

import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio.v2 as imageio
import cartopy.crs as ccrs
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import seaborn as sns
```


```python
sns.set_theme(style="whitegrid", context="paper")

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False
})
```


```python
# --- EPA ---
epa = load_epa_data()
epa_june = filter_june(epa)

gdf = gpd.GeoDataFrame(
    epa_june,
    geometry=gpd.points_from_xy(epa_june["longitude"], epa_june["latitude"]),
    crs="EPSG:4326"
)

# --- FINN ---
finn = load_finn_june_processed()
pm = finn["pm_tons"]
lat = finn["lat"].values
lon = finn["lon"].values
# time = finn["time"].values
time = pd.to_datetime(finn["time"].values)
pm = pm.chunk({'time': -1, 'lat': 500, 'lon': 500})
```

    /glade/u/home/spatrin/src/data_loading.py:42: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.
      ds = xr.open_dataset(



```python
ga_mask = get_georgia_mask(lat, lon)
ga_mask = ga_mask.astype(np.bool_)
```


```python
@njit(parallel=True)

def apply_mask_fast(grid, mask):
    for i in prange(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not mask[i, j]:
                grid[i, j] = np.nan
    return grid

@njit
def compute_miss_fast(epa_grid, finn_grid):
    miss = np.zeros_like(epa_grid)

    for i in range(epa_grid.shape[0]):
        for j in range(epa_grid.shape[1]):
            if epa_grid[i, j] > 0 and finn_grid[i, j] <= 1e-3:
                miss[i, j] = 1

    return miss
```


```python
epa_dates = pd.to_datetime(gdf["date"].values).floor("D").values
epa_lat = gdf.geometry.y.values
epa_lon = gdf.geometry.x.values
epa_type = gdf["type"].values
```


```python
unique_dates, inverse_idx = np.unique(epa_dates, return_inverse=True)

date_groups = [
    np.where(inverse_idx == i)[0]
    for i in range(len(unique_dates))
]
```


```python
finn_times = pd.to_datetime(time).floor("D").values

time_lookup = np.zeros(len(unique_dates), dtype=np.int32)

for i, d in enumerate(unique_dates):
    time_lookup[i] = np.argmin(np.abs(finn_times - d))
```


```python
dlat = lat[1] - lat[0]
dlon = lon[1] - lon[0]

lat_edges = np.concatenate([[lat[0] - dlat/2], lat + dlat/2])
lon_edges = np.concatenate([[lon[0] - dlon/2], lon + dlon/2])
```


```python
nlat = len(lat)
nlon = len(lon)

miss_ag = np.zeros((nlat, nlon))
miss_prescribed = np.zeros((nlat, nlon))
miss_wildfire = np.zeros((nlat, nlon))
miss_grid_total = np.zeros((nlat, nlon))
```


```python
for i in range(len(unique_dates)):

    inds = date_groups[i]

    if len(inds) == 0:
        continue

    # --- subset arrays ---
    lat_pts = epa_lat[inds]
    lon_pts = epa_lon[inds]
    types = epa_type[inds]

    epa_grid, _, _ = np.histogram2d(
        lat_pts,
        lon_pts,
        bins=[lat_edges, lon_edges]
    )
    epa_grid = apply_mask_fast(epa_grid, ga_mask)

    finn_idx = time_lookup[i]
    finn_grid = pm.isel(time=finn_idx).values
    finn_grid = apply_mask_fast(finn_grid, ga_mask)

    miss = compute_miss_fast(epa_grid, finn_grid)

    ag_mask = types == "Agricultural"
    pr_mask = types == "Prescribed"
    wf_mask = types == "Wildfire"

    if np.any(ag_mask):
        ag_grid, _, _ = np.histogram2d(
            lat_pts[ag_mask], lon_pts[ag_mask],
            bins=[lat_edges, lon_edges]
        )
        miss_ag += (ag_grid > 0) * miss
    if np.any(pr_mask):
        pr_grid, _, _ = np.histogram2d(
            lat_pts[pr_mask], lon_pts[pr_mask],
            bins=[lat_edges, lon_edges]
        )
        miss_prescribed += (pr_grid > 0) * miss

    if np.any(wf_mask):
        wf_grid, _, _ = np.histogram2d(
            lat_pts[wf_mask], lon_pts[wf_mask],
            bins=[lat_edges, lon_edges]
        )
        miss_wildfire += (wf_grid > 0) * miss

    miss_grid_total += miss
```


```python
print("Miss total sum:", np.nansum(miss_grid_total))
print("AG sum:", np.nansum(miss_ag))
print("Prescribed sum:", np.nansum(miss_prescribed))
print("Wildfire sum:", np.nansum(miss_wildfire))
```

    Miss total sum: 1623.0
    AG sum: 848.0
    Prescribed sum: 468.0
    Wildfire sum: 343.0



```python
cmap = plt.cm.Reds

cmap_trim = mcolors.LinearSegmentedColormap.from_list(
    "Reds_trim",
    cmap(np.linspace(0.3, 1, 256))
)
```


```python
miss_ag[miss_ag == 0] = np.nan
miss_prescribed[miss_prescribed == 0] = np.nan
miss_wildfire[miss_wildfire == 0] = np.nan
miss_grid_total[miss_grid_total == 0] = np.nan
```


```python
vmax = np.nanpercentile(miss_grid_total, 99)
```


```python
pm_np = pm.values  # (time, lat, lon)
pm_np = np.where(ga_mask, pm_np, np.nan)
```


```python
print("EPA lon range:", gdf.geometry.x.min(), gdf.geometry.x.max())
print("EPA lat range:", gdf.geometry.y.min(), gdf.geometry.y.max())
print("FINN lon range:", lon.min(), lon.max())
print("FINN lat range:", lat.min(), lat.max())
```

    EPA lon range: -85.57116111 -81.14180556
    EPA lat range: 30.5301289 34.97691667
    FINN lon range: -179.95001 179.95001
    FINN lat range: -89.95 89.850006


## Plots


```python
ds = 5

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

datasets = [
    ("Total Miss", miss_grid_total),
    ("Agricultural", miss_ag),
    ("Prescribed", miss_prescribed),
    ("Wildfire", miss_wildfire),
]

for ax, (title, data) in zip(axes.ravel(), datasets):
    basemap(ax)

    m = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=cmap_trim
    )

    ax.set_title(title)
    plt.colorbar(m, ax=ax, shrink=0.8)

plt.tight_layout()

```


```python
epa_grid_total = np.zeros((nlat, nlon))

for i in range(len(unique_dates)):
    inds = date_groups[i]

    if len(inds) == 0:
        continue

    lat_pts = epa_lat[inds]
    lon_pts = epa_lon[inds]

    grid, _, _ = np.histogram2d(
        lat_pts,
        lon_pts,
        bins=[lat_edges, lon_edges]
    )

    grid = apply_mask_fast(grid, ga_mask)

    epa_grid_total += grid

epa_grid_total[epa_grid_total == 0] = np.nan
```


```python
finn_active_total = np.zeros((nlat, nlon))
threshold = 1e-3

for i in range(len(unique_dates)):
    finn_idx = time_lookup[i]
    finn_grid = pm_np[finn_idx]
    finn_active_total += (finn_grid > threshold)

finn_active_total[finn_active_total <= 0] = np.nan

vmax = np.nanpercentile(finn_active_total, 99)

fig, ax = basemap()

mesh = ax.pcolormesh(
    lon,
    lat,
    finn_active_total,
    cmap="Purples",
    vmin=0,
    vmax=vmax,
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(mesh, ax=ax, label="FINN active cells (days)")
ax.set_title("FINN Detection Activity (June)")

plt.show()
```


```python
tp_grid_total = np.zeros((nlat, nlon))
threshold = 1e-3

for i in range(len(unique_dates)):

    inds = date_groups[i]
    if len(inds) == 0:
        continue

    # --- EPA grid ---
    lat_pts = epa_lat[inds]
    lon_pts = epa_lon[inds]

    epa_grid, _, _ = np.histogram2d(
        lat_pts,
        lon_pts,
        bins=[lat_edges, lon_edges]
    )

    epa_grid = apply_mask_fast(epa_grid, ga_mask)

    finn_idx = time_lookup[i]
    finn_grid = pm_np[finn_idx]

    tp = (epa_grid > 0) & (finn_grid > threshold)
    tp_grid_total += tp

tp_grid_total[tp_grid_total <= 0] = np.nan
miss_grid_total[miss_grid_total <= 0] = np.nan

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

datasets = [
    ("Correct Detections", tp_grid_total, "Greens"),
    ("Missed Fires", miss_grid_total, cmap_trim)
]

for ax, (title, grid, cmap_use) in zip(axes, datasets):

    basemap(ax)

    mesh = ax.pcolormesh(
        lon,
        lat,
        grid,
        cmap=cmap_use,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    ax.set_title(title)
    plt.colorbar(mesh, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correct_vs_missed.png"), dpi=300)

plt.show()
```


```python
epa_active = miss_grid_total > 0   # EPA fires exist
finn_active = (pm.sum(dim="time").values > 0)  # FINN detection

category = np.zeros((nlat, nlon))

category[epa_active & (~finn_active)] = 1   # EPA only
category[(~epa_active) & finn_active] = 2   # FINN only
category[epa_active & finn_active] = 3      # both
```


```python
category_plot = category.copy()
category_plot = np.where(ga_mask, category_plot, np.nan)
category_plot[category_plot == 0] = np.nan

fig, ax = basemap()

cmap = ListedColormap([
    "#FFFFFF",  # no data
    "#8B0000",  # EPA only
    "#E6B800",  # FINN only
    "#2E7D32",  # both
])

mesh = ax.pcolormesh(
    lon,
    lat,
    category_plot,
    cmap=cmap,
    vmin=0,
    vmax=3)

ax.set_title("Spatial Agreement of Fire Detections (June 2022)")

legend_patches = [
    mpatches.Patch(color="#8B0000", label="EPA only"),
    mpatches.Patch(color="#E6B800", label="FINN only"),
    mpatches.Patch(color="#2E7D32", label="Both"),
]

ax.legend(handles=legend_patches, loc="lower left")
fname = os.path.join(FIG_DIR, "detection_agreement.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.show()
```


```python
epa_count = (miss_grid_total > 0).astype(int)
finn_count = (pm.sum(dim="time").values > 0).astype(int)
both_count = epa_count & finn_count
```


```python
agreement_ratio = np.zeros_like(both_count, dtype=float)

valid = epa_count > 0
agreement_ratio[valid] = both_count[valid] / epa_count[valid]
agreement_ratio[~valid] = np.nan
agreement_ratio[agreement_ratio < 0] = np.nan
```


```python
fig, ax = basemap()

mesh = ax.pcolormesh(
    lon,
    lat,
    agreement_ratio,
    cmap=cmap_trim,
    vmin=0,
    vmax=1,
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(mesh, ax=ax, label="Detection Agreement Rate")

ax.set_title("FINN Detection Success Rate (per grid cell)")

plt.show()
```


```python
fig, ax = basemap()

mesh = ax.pcolormesh(
    lon,
    lat,
    epa_count,
    cmap="Blues",
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(mesh, ax=ax, label="EPA fire days")
ax.set_title("EPA Fire Occurrence Frequency")

plt.show()
```


```python
fig, ax = basemap()

mesh = ax.pcolormesh(
    lon,
    lat,
    finn_count,
    cmap="Purples",
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(mesh, ax=ax, label="FINN active days")
ax.set_title("FINN Detection Frequency")

plt.show()
```


```python
epa_type = gdf["type"].values
epa_area = gdf["area"].values
epa_lat = gdf.geometry.y.values
epa_lon = gdf.geometry.x.values
epa_dates = pd.to_datetime(gdf["date"]).dt.floor("D").values

finn_times = pd.to_datetime(finn["time"].values).floor("D").values

time_lookup = {
    d: np.argmin(np.abs(finn_times - d))
    for d in np.unique(epa_dates)
}

matched = np.zeros(len(gdf), dtype=np.int8)

for d in np.unique(epa_dates):

    inds = np.where(epa_dates == d)[0]

    if len(inds) == 0:
        continue

    # --- FINN GRID ---
    finn_idx = time_lookup[d]
    finn_grid = pm.isel(time=finn_idx).values

    finn_grid = np.where(ga_mask, finn_grid, np.nan)

    lat_pts = epa_lat[inds]
    lon_pts = epa_lon[inds]

    lat_idx = np.abs(lat[:, None] - lat_pts).argmin(axis=0)
    lon_idx = np.abs(lon[:, None] - lon_pts).argmin(axis=0)

    vals = finn_grid[lat_idx, lon_idx]

    matched[inds] = (vals > 1e-3).astype(np.int8)

epa_classified = pd.DataFrame({
    "type": epa_type,
    "area": epa_area,
    "matched": matched,
    "lat": epa_lat,
    "lon": epa_lon
})

epa_classified["missed"] = 1 - epa_classified["matched"]
```


```python
type_summary = (
    epa_classified
    .groupby("type", observed=False)
    .agg(
        missed=("missed", "sum"),
        total=("matched", "count")
    )
)

type_summary["miss_percent"] = (
    type_summary["missed"] / type_summary["total"] * 100
)
```


```python
order = ["Prescribed", "Agricultural", "Wildfire"]
type_summary = type_summary.reindex(order)

fire_colors = ["red", "gold", "orange"]

plt.figure(figsize=(10, 7))

bars = plt.bar(
    type_summary.index,
    type_summary["miss_percent"],
    color=fire_colors,
    edgecolor="black"
)

# labels
for i, (v, n) in enumerate(zip(type_summary["miss_percent"], type_summary["total"])):
    plt.text(i, v + 1, f"{v:.1f}%\n(n={n})", ha="center")

plt.ylabel("Missed Fires (%)", fontsize=8)
plt.title("Detection Performance by Fire Type\nGeorgia, June 2022", fontsize=10)

plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)
fname = os.path.join(FIG_DIR, "detection_performance.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.show()
```


```python
bins = [0, 1, 10, 50, 100, 500, np.inf]
labels = ["<1", "1–10", "10–50", "50–100", "100–500", "500+"]

epa_classified["size_bin"] = pd.cut(
    epa_classified["area"],
    bins=bins,
    labels=labels
)

size_summary = (
    epa_classified
    .groupby("size_bin", observed=False)
    .agg(
        missed=("missed", "sum"),
        total=("matched", "count")
    )
)

size_summary["miss_percent"] = (
    size_summary["missed"] / size_summary["total"] * 100
)
```


```python
print(epa_classified["matched"].mean())
```


```python
x_labels = size_summary.index.astype(str)

plt.figure(figsize=(5.5, 4))

plt.plot(
    x_labels,
    size_summary["miss_percent"],
    marker="o",
    color="black",
    linewidth=1.5
)

plt.scatter(
    x_labels,
    size_summary["miss_percent"],
    color="gray",
    s=50,
    zorder=3
)

plt.ylabel("Missed Fires (%)")
plt.xlabel("Fire Size (acres)")
plt.title("Detection Bias by Fire Size")

plt.ylim(0, 100)
plt.grid(alpha=0.3)
fname = os.path.join(FIG_DIR, "detection_bias.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.show()
```


```python
# --- APPLY GA MASK ---
lat_idx = np.abs(lat[:, None] - epa_classified["lat"].values).argmin(axis=0)
lon_idx = np.abs(lon[:, None] - epa_classified["lon"].values).argmin(axis=0)

inside_mask = ga_mask[lat_idx, lon_idx]
epa_classified = epa_classified.loc[inside_mask].copy()


# --- SPLIT ---
detected = epa_classified[epa_classified["matched"] == 1]
missed = epa_classified[epa_classified["matched"] == 0]


# --- PLOT ---
fig, ax = basemap()

ax.scatter(missed["lon"], missed["lat"], color="red", s=6, alpha=0.3, label="Missed", transform=ccrs.PlateCarree(), zorder=3)
ax.scatter(detected["lon"], detected["lat"], color="blue", s=10, alpha=0.9, label="Detected", transform=ccrs.PlateCarree(), zorder=4)

ax.legend()
ax.set_title("EPA Fires: Detected vs Missed (Georgia, June 2022)")
plt.show()
```

## plot for slide


```python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
epa_grid = epa_to_finn_grid(
    gdf,
    lat,
    lon,
    weight_type="pm25"
)

epa_grid = apply_ga_mask(epa_grid, ga_mask)

plot_grid = epa_grid.copy()
plot_grid[plot_grid <= 0] = np.nan

cmap_trim = mcolors.LinearSegmentedColormap.from_list(
    "Reds_trim",
    plt.cm.Reds(np.linspace(0.3, 1, 256))
)

# log scaling
vmin = np.nanpercentile(plot_grid, 1)
vmax = np.nanpercentile(plot_grid, 99)

norm = colors.LogNorm(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

fig.subplots_adjust(
    left=0.05,
    right=0.9,
    bottom=0.08,
    top=0.87,
    wspace=0.02
)

# --- EPA POINTS ---
ax = axes[0]
basemap(ax)

vals = gdf["pm2.5"].fillna(0).values
vals_sqrt = np.sqrt(vals)

sizes = 5 + (vals_sqrt / (np.nanmax(vals_sqrt) + 1e-6)) * 80

ax.scatter(
    gdf.geometry.x,
    gdf.geometry.y,
    s=sizes,
    color="#cb181d",
    edgecolors=None,
    alpha=0.6,
    transform=ccrs.PlateCarree(),
    zorder=4
)

axes[0].text(
    -0.03, 1.02,
    "A",
    transform=axes[0].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top"
)

axes[1].text(
    -0.03, 1.02,
    "B",
    transform=axes[1].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top"
)

axes[0].set_title("EPA Fire Locations", fontsize=12, pad=8)
axes[1].set_title("EPA PM$_{2.5}$ Emissions Grid", fontsize=12, pad=8)

# --- EPA GRID ---
ax = axes[1]
basemap(ax)

mesh = ax.pcolormesh(
    lon,
    lat,
    plot_grid,
    cmap=cmap_trim,
    norm=norm,
    shading="nearest",
    zorder=4)


cax = inset_axes(
    axes[1],
    width="4%",
    height="65%",
    loc="center right",
    bbox_to_anchor=(0.05, 0., 1, 1),
    bbox_transform=axes[1].transAxes,
    borderpad=0
)

cbar = fig.colorbar(mesh, cax=cax)
cbar.set_label(
    "PM$_{2.5}$ (tons)",
    fontsize=11
)
cbar.ax.tick_params(labelsize=10)

# plt.subplots_adjust(hspace=0.25)

fig.suptitle(
    "EPA Fire Emissions in Georgia, June 2022",
    fontsize=16,
    fontweight="bold",
    y=0.96
)

fname = os.path.join(FIG_DIR, "epa_pt_grid.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.show()
```


    
![png](output_39_0.png)
    



```python

```
