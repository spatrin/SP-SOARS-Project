"""
"""

import pandas as pd
import xarray as xr
import numpy as np
from utils import DATA_DIR, POLLUTANTS


def load_epa_data():
    """
    Load and combine EPA datasets.
    """

    epa_ag = pd.read_csv(f"{DATA_DIR}/2022FireLoc_Georgia_ag.csv")
    wfx = pd.read_csv(f"{DATA_DIR}/2022v2_FireLoc_Georgia_wf_rx.csv")


    wf = wfx[wfx["type"] == "WF"]
    rx = wfx[wfx["type"] == "RX"]

    df = pd.concat([
        epa_ag.assign(type="Agricultural"),
        wf.assign(type="Wildfire"),
        rx.assign(type="Prescribed")
    ], ignore_index=True, copy=False)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["month"] = df["date"].dt.month

    cols = [c for c in POLLUTANTS if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    return df


def load_finn_pm25():
    """
    Load FINN PM2.5 dataset.
    """

    ds = xr.open_dataset(
        f"{DATA_DIR}/emissions-finnv2.5modvrs_PM25_bb_surface_daily_20220101-20221231_0.1x0.1.nc",
        chunks={"time": 1}
    )

    # fix longitude
    lon = (ds.lon + 180) % 360 - 180
    ds = ds.assign_coords(lon=lon).sortby("lon")

    pm = ds["fire_modisviirs_PM25"]

    return {
        "pm": pm,
        "lat": ds.lat,
        "lon": ds.lon,
        "time": pd.to_datetime(ds.time.values)
    }

def filter_june(df):
    """
    Filter dataset to June.
    """
    return df[df["month"] == 6].copy()