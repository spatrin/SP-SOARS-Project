"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Datasets")

FIRE_TYPES = ["Agricultural", "Prescribed", "Wildfire"]

FIRE_COLORS = {
    "Prescribed": "red",
    "Agricultural": "gold",
    "Wildfire": "orange"
}

POLLUTANTS = [
    "pm2.5", "pm10", "co", "co2",
    "ch4", "nox", "nh3", "so2", "voc"
]

MONTHS_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def enforce_fire_order(series):
    """
    Ensure consistent ordering of fire types.
    """
    order = ["Prescribed", "Agricultural", "Wildfire"]
    return series.reindex(order)


def get_fire_colors(order):
    """
    Return colors in correct order.
    """
    return [FIRE_COLORS[t] for t in order]


def add_bar_labels(ax, values, offset=20):
    values = np.asarray(values)
    for i, v in enumerate(values):
        ax.text(i, v + offset, f"{int(v)}", ha="center")


def log_transform(series):
    return np.log10(np.asarray(series) + 1e-6)


def cumulative_curve(values):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)[::-1]

    total = vals.sum()
    if total == 0:
        return np.array([]), np.array([])

    cumsum = np.cumsum(vals) / total * 100
    x = np.arange(len(vals)) / len(vals) * 100

    return x, cumsum