import numpy as np
import geopandas as gpd
from shapely.vectorized import contains

_states = None
_counties = None

def get_georgia_mask(lat, lon):
    global _states

    if _states is None:
        _states = gpd.read_file(
            "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
        )

    ga = _states[_states["NAME"] == "Georgia"].geometry.values[0]

    # create 2D mesh grid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # vectorized point-in-polygon
    mask = contains(ga, lon2d, lat2d)

    return mask


def apply_ga_mask(grid, mask):
    return np.where(mask, grid, np.nan)

def get_states():
    global _states
    if _states is None:
        _states = gpd.read_file(
            "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
        )
    return _states

def get_ga_counties():
    global _counties
    if _counties is None:
        _counties = gpd.read_file(
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        )
    return _counties[_counties["id"].str.startswith("13")]
