"""
"""
import numpy as np
import pandas as pd
from spatial_utils import (
    epa_to_finn_grid,
    compute_overlap_metrics,
    spatial_correlation
)
from data_loading import load_finn_pm25
from conversion import finn_flux_to_tons_per_day
from aggregation import compute_june, compute_daily_total, compute_total_map
from geo_utils import *


def load_finn_june_processed():

    data = load_finn_pm25()

    pm = data["pm"]
    lat = data["lat"]
    lon = data["lon"]

    pm_june = compute_june(pm)

    pm_tons = finn_flux_to_tons_per_day(pm_june, lat, lon)

    pm_daily = compute_daily_total(pm_tons)
    pm_total = compute_total_map(pm_tons)

    return {
        "pm_tons": pm_tons,     # (time, lat, lon)
        "lat": lat,
        "lon": lon,
        "time": pm_june.time,
        "pm_june": pm_june,
        "pm_daily": pm_daily,
        "pm_total": pm_total
    }

def load_finn_annual_processed():

    data = load_finn_pm25()

    pm_annual = data["pm"]
    lat = data["lat"]
    lon = data["lon"]

    pm_tons = finn_flux_to_tons_per_day(pm_annual, lat, lon)

    pm_daily = compute_daily_total(pm_tons)
    pm_total = compute_total_map(pm_tons)

    return {
        "pm_tons": pm_tons,     # (time, lat, lon)
        "lat": lat,
        "lon": lon,
        "time": pm_annual.time,
        "pm_daily": pm_daily,
        "pm_total": pm_total
    }

def run_daily_spatiotemporal_analysis(epa_gdf, finn, lat, lon):

    daily_results = []
    TP_total = 0
    FN_total = 0
    FP_total = 0

    dates = sorted(pd.to_datetime(epa_gdf["date"]).dt.normalize().unique())
    mask = get_georgia_mask(lat, lon)

    for date in dates:

        # --- 1. subset EPA ---
        epa_day = epa_gdf[
            pd.to_datetime(epa_gdf["date"]).dt.normalize() == date
        ]

        if len(epa_day) == 0:
            continue

        # --- 2. map to grid ---
        epa_grid = epa_to_finn_grid(epa_day, lat, lon, weight_type="count")

        # --- 3. subset FINN ---
        finn_da = finn["pm_tons"]

        finn_day = finn_da.sel(time=date, method="nearest")
        finn_grid = np.array(finn_day)

        epa_grid  = apply_ga_mask(epa_grid, mask)
        finn_grid = apply_ga_mask(finn_grid, mask)

        # --- 4. compute metrics ---
        metrics = compute_overlap_metrics(epa_grid, finn_grid)

        daily_results.append({
            "date": date,
            "TP": metrics["TP"],
            "FN": metrics["FN"],
            "FP": metrics["FP"],
            "TN": metrics.get("TN", np.nan)
        })

        TP_total += metrics["TP"]
        FN_total += metrics["FN"]
        FP_total += metrics["FP"]


    daily_results = pd.DataFrame(daily_results)

    daily_results["miss_rate"] = (daily_results["FN"] / (daily_results["TP"] + daily_results["FN"]))

    return {
        "daily": daily_results,
        "TP": TP_total,
        "FN": FN_total,
        "FP": FP_total,
        "precision": TP_total / (TP_total + FP_total + 1e-6),
        "recall": TP_total / (TP_total + FN_total + 1e-6),
    }