"""
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from geo_utils import *

def build_kdtree(lat, lon):
    """
    Build KD-tree from grid.
    """
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    tree = cKDTree(grid_points)
    return tree, lat_mesh


def match_point(lat, lon, tree, grid_shape):
    """
    Match a point to nearest grid cell.
    """
    dist, idx = tree.query([lat, lon])
    lat_idx, lon_idx = np.unravel_index(idx, grid_shape)
    return lat_idx, lon_idx, dist * 111


def epa_to_finn_grid(epa_gdf, lat, lon, weight_type="pm25"):
    """
    Map EPA point data onto FINN grid.

    Parameters:
        epa_gdf : GeoDataFrame with geometry + emissions
        lat, lon : FINN grid centers
        weight_type : "count", "pm25", or "area"

    Returns:
        grid : 2D array (lat, lon)
    """

    # --- grid spacing ---
    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])

    # --- edges (center ± half spacing) ---
    lat_edges = np.concatenate([[lat[0] - dlat/2], lat + dlat/2])
    lon_edges = np.concatenate([[lon[0] - dlon/2], lon + dlon/2])

    # --- weights ---
    if weight_type == "count":
        weights = None

    elif weight_type == "pm25":
        weights = epa_gdf["pm2.5"].fillna(0).values

    elif weight_type == "area":
        weights = epa_gdf["area"].fillna(0).values

    else:
        raise ValueError("weight_type must be 'count', 'pm25', or 'area'")

    # --- 2D histogram ---
    grid, _, _ = np.histogram2d(
        epa_gdf.geometry.y,   # lat
        epa_gdf.geometry.x,   # lon
        bins=[lat_edges, lon_edges],
        weights=weights
    )

    return grid

def classify_missed_fires(epa_gdf, finn_grid, lat, lon, threshold=0):
    """
    Classify EPA fires as matched (detected) or missed by FINN.

    threshold : minimum FINN value to count as detection
    """

    tree, lat_mesh = build_kdtree(lat, lon)

    coords = np.column_stack([
        epa_gdf.geometry.y,
        epa_gdf.geometry.x
    ])

    _, idx = tree.query(coords)

    lat_idx, lon_idx = np.unravel_index(idx, lat_mesh.shape)

    # apply threshold
    matched = finn_grid[lat_idx, lon_idx] > threshold

    epa_gdf = epa_gdf.copy()
    epa_gdf["matched"] = matched.astype(int)

    return epa_gdf

def compute_overlap(epa_grid, finn_grid):
    """
    Compute overlap between EPA and FINN active cells.
    """
    overlap = (epa_grid > 0) & (finn_grid > 0)
    epa_only = (epa_grid > 0) & (finn_grid == 0)
    finn_only = (epa_grid == 0) & (finn_grid > 0)
    return overlap, epa_only, finn_only


def normalize_grid(grid):
    max_val = np.nanmax(grid)
    return grid / max_val if max_val > 0 else grid


def log_transform_grid(grid):
    """
    Log transform grid.
    """
    return np.log10(grid + 1)

def spatial_correlation(epa_grid, finn_grid):
    """
    Compute log-space spatial correlation between EPA and FINN.

    Only considers overlapping, nonzero cells.
    """

    valid = (
        (epa_grid > 0) &
        (finn_grid > 0) &
        np.isfinite(epa_grid) &
        np.isfinite(finn_grid)
    )

    if np.sum(valid) < 5:   # lower threshold
        return np.nan

    x = np.log10(epa_grid[valid] + 1)
    y = np.log10(finn_grid[valid] + 1)
    return np.corrcoef(x, y)[0, 1]


def basemap(ax=None):
    """
    Georgia basemap with consistent styling.
    """

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    else:
        fig = None

    extent = [-85.7, -80.5, 30.3, 35.2]
    ax.set_extent(extent)

    # Water (soft desaturated blue)
    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#dfe7ef",
        zorder=0
    )
    ax.add_feature(
        cfeature.LAKES,
        facecolor="#dfe7ef",
        edgecolor="#9aa7b3",
        linewidth=0.5,
        zorder=3
    )
    ax.add_feature(
        cfeature.RIVERS,
        edgecolor="#9aa7b3",
        linewidth=0.6,
        zorder=3
    )

    # Land base
    ax.add_feature(
        cfeature.LAND,
        facecolor="#ffffff",
        zorder=0
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    
    # States
    states = get_states()
    ga = states[states["NAME"] == "Georgia"]
    other_states = states[states["NAME"] != "Georgia"]

    # Surrounding states (faint fill)
    ax.add_geometries(
        other_states.geometry,
        ccrs.PlateCarree(),
        facecolor="#eeeeee",
        edgecolor="#c7c7c7",
        linewidth=0.5,
        zorder=1
    )

    # Georgia highlight (neutral gray emphasis)
    ax.add_geometries(
        ga.geometry,
        ccrs.PlateCarree(),
        facecolor="#e6e6e6",
        edgecolor="#4d4d4d",
        linewidth=1.3,
        zorder=2
    )

    # Counties (thin, subtle)
    ga_counties = get_ga_counties()
    ax.add_geometries(
        ga_counties.geometry,
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='#8c8c8c',
        linewidth=0.2,
        alpha=0.3,
        zorder=3
    )

    # Cities
    cities = {
        "Atlanta": (-84.39, 33.75),
        "Savannah": (-81.10, 32.08),
        "Athens": (-83.38, 33.95),
        "Macon": (-83.63, 32.84),
        "Augusta": (-81.97, 33.47)
    }
    for city, (lon, lat) in cities.items():
        ax.scatter(
            lon, lat,
            s=20,
            color="#ffffff",
            edgecolor="#333333",
            linewidth=0.8,
            zorder=7
        )
    
        ax.text(
            lon + 0.1,
            lat + 0.05,
            city,
            fontsize=9,
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    ax.axis("off")
    
    # ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
    # ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

    # ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    # # OPTIONAL: counties (if available)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # # ax.outline_patch.set_linewidth(1.2)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color='gray',
        alpha=0.3
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    gl.xlocator = mticker.FixedLocator([-85, -84, -83, -82, -81])
    gl.ylocator = mticker.FixedLocator([31, 32, 33, 34, 35])

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    if fig is not None:
        return fig, ax
    return ax


def plot_finn_epa_overlay(pm_data, epa_grid, lon, lat, date=None, norm=None, ax=None):

    if ax is None:
        fig, ax = basemap()
    else:
        fig = None

    # mask invalid
    pm_data = pm_data.copy()
    pm_data[pm_data <= 0] = np.nan

    # norm
    if norm is None:
        valid = pm_data[(pm_data > 0) & np.isfinite(pm_data)]

        if len(valid) == 0:
            vmin, vmax = 1e-3, 1
        else:
            vmin = np.percentile(valid, 5)
            vmax = np.percentile(valid, 99)

        if vmin >= vmax:
            vmax = vmin * 10

        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # FINN layer
    cmap = plt.get_cmap("YlOrRd")
    
    mesh = ax.pcolormesh(
        lon, lat, pm_data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=2
    )

    # EPA grid overlay
    ax.pcolormesh(
        lon, lat, epa_grid,
        cmap="Blues",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4
    )

    if date is not None:
        ax.set_title(
            f"FINN PM$_{{2.5}}$ (tons) + EPA Fires\n{date.strftime('%B %d, %Y')}"
        )

    return mesh


def get_daily_epa(df, date):
    date = pd.Timestamp(date).normalize()

    return df[
        pd.to_datetime(df["date"]).dt.normalize() == date
    ]

def compute_overlap_metrics(epa_grid, finn_grid):
    """
    Compute detection metrics at grid level.
    """

    epa_active = epa_grid > 0
    finn_active = finn_grid > 0

    TP = np.sum(epa_active & finn_active)
    FN = np.sum(epa_active & ~finn_active)
    FP = np.sum(~epa_active & finn_active)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "precision": precision,
        "recall": recall
    }