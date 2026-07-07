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
# from utils import log_transform
# from data_loading import load_finn_pm25
# from data_loading import load_finn_june
# from conversion import finn_to_tons
from spatial_utils import basemap
from pipeline import load_finn_june_processed
```


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
```


```python
# data = load_finn_june()
data = load_finn_june_processed()

# pm_june = data["pm_june"]
# pm_total_tons = data["pm_total_tons"]
# pm_daily_tons = data["pm_daily_tons"]

# lat = data["lat"]
# lon = data["lon"]
# time = data["time"]


pm_june = data["pm_june"]
pm_tons = data["pm_tons"]
pm_total_tons = data["pm_total_tons"]
pm_daily_tons = data["pm_daily_tons"]
lat = data["lat"]
lon = data["lon"]
time = data["time"]
```

    /glade/u/home/spatrin/src/data_loading.py:42: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.
      ds = xr.open_dataset(



```python
print(f"Grid shape: {pm_june.shape}")
print(f"Latitude range: {float(lat.min()):.2f} to {float(lat.max()):.2f}")
print(f"Longitude range: {float(lon.min()):.2f} to {float(lon.max()):.2f}")
print(f"Time range: {str(time.min().values)} to {str(time.max().values)}")
```

    Grid shape: (30, 1799, 3600)
    Latitude range: -89.95 to 89.85
    Longitude range: -179.95 to 179.95
    Time range: 2022-06-01T00:00:00.000000000 to 2022-06-30T00:00:00.000000000



```python
values = pm_total_tons.values
values = values[np.isfinite(values)].flatten()

THRESH = np.percentile(values[values > 0], 25)

total_cells = len(values)
active_cells = np.sum(values > THRESH)

print(f"Total cells: {total_cells}")
print(f"Active cells (> threshold): {active_cells}")
print(f"Active fraction: {100 * active_cells / total_cells:.2f}%")
```

    Total cells: 6476400
    Active cells (> threshold): 72398
    Active fraction: 1.12%



```python
vals = values[values > THRESH]
vals_log = np.log10(vals)

plt.figure()

plt.hist(vals_log, bins=100)

plt.title("FINN PM2.5 Distribution (log10)")
plt.xlabel("log10(PM2.5 tons)")
plt.ylabel("Frequency")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_6_0.png)
    



```python
detected_cells = np.sum(values > 0)
detected_fraction = detected_cells / total_cells

print(f"Detected cells (>0): {detected_cells}")
print(f"Detection coverage: {100 * detected_fraction:.2f}%")
```

    Detected cells (>0): 96531
    Detection coverage: 1.49%


FINN PM2.5 emissions exhibit strong spatial concentration, with only ~1–2% 
of grid cells exceeding the 25th percentile of nonzero emissions. This indicates that fire activity is highly localized, with a small number 
of grid cells contributing the majority of total emissions. The distribution is strongly right-skewed, spanning several orders of magnitude, 
which reflects the dominance of a few large fires over many smaller events.


```python
pm_daily_tons = data["pm_daily_tons"]
plt.figure()

plt.plot(pm_daily_tons.values, color="black", linewidth=2)

plt.title("Daily FINN PM2.5 Emissions (June 2022)")
plt.xlabel("Day")
plt.ylabel("Total PM2.5 (tons)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```


    
![png](output_9_0.png)
    



```python
flat_vals = pm_total_tons.values
flat_vals = flat_vals[np.isfinite(flat_vals)]
flat_vals = flat_vals[flat_vals > THRESH]

top_vals = np.sort(flat_vals)[::-1]

plt.figure()
plt.plot(np.log10(top_vals[:200]))

plt.title("Top FINN Grid Cell Emissions")
plt.xlabel("Rank")
plt.ylabel("log10(PM2.5 tons)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```


    
![png](output_10_0.png)
    


A small number of grid cells dominate total emissions, indicating a highly skewed spatial distribution.


```python
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter

fig, ax = basemap()

pm_plot = pm_total_tons.values.copy()
pm_plot[pm_plot <= 1e-6] = np.nan

norm = LogNorm(
    vmin=np.nanpercentile(pm_plot, 1),
    vmax=np.nanpercentile(pm_plot, 99)
)

mesh = ax.pcolormesh(
    lon,
    lat,
    pm_plot,
    cmap="YlOrRd",
    norm=norm,
    transform=ccrs.PlateCarree(),
    shading="nearest",
    zorder=2
)

cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, extend='neither')
mesh.set_clim(norm.vmin, norm.vmax)

log_min = np.log10(norm.vmin)
log_max = np.log10(norm.vmax)
tick_logs = np.linspace(log_min, log_max, 4)
ticks = 10**tick_logs

cbar.set_ticks(ticks)

cbar.set_ticklabels([rf"$10^{{{int(np.log10(t))}}}$" for t in ticks])
cbar.ax.minorticks_off()

cbar.set_label(r"PM$_{2.5}$ emissions (tons)")

ax.set_title("FINN PM$_{2.5}$ Emissions (June 2022)", fontsize=12)

plt.tight_layout()

fname = os.path.join(FIG_DIR, "FINN_pm25_emissions.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=True
)
plt.show()
```


    
![png](output_12_0.png)
    


1. FINN emissions show substantially higher spatial coverage when using MODIS+VIIRS
2. Emissions are still highly skewed, with a small number of grid cells contributing most of the total
3. Daily emissions vary over time but remain concentrated in specific regions
4. The spatial structure is now more continuous and better reflects known fire activity
5. This improved detection will likely increase overlap with EPA observations


```python

```
