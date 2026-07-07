"""
"""
def compute_june(pm):
    return pm.sel(time=pm.time.dt.month == 6)


def compute_daily_total(pm_tons):
    """
    (time, lat, lon) → (time,)
    """
    return pm_tons.sum(dim=["lat", "lon"])


def compute_total_map(pm_tons):
    """
    (time, lat, lon) → (lat, lon)
    """
    return pm_tons.sum(dim="time")