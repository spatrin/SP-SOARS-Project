import numpy as np

 # --- constants ---
AVOGADRO = 6.022e23  # molecules/mol
MOLAR_MASS = 12        # g/mol
CM2_TO_M2 = 1e4        # cm^2 → m^2
G_TO_TONS = 1 / 907185
R = 6371000            # Earth radius (meters)

def gridcell_area(lat, lon):
    """
    Compute grid cell area (m²)
    """

    lat = lat.values if hasattr(lat, "values") else lat
    lon = lon.values if hasattr(lon, "values") else lon

    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])

    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    lat_rad = np.deg2rad(lat)

    lat_n = lat_rad + dlat_rad / 2
    lat_s = lat_rad - dlat_rad / 2

    area = (
        R**2
        * (np.sin(lat_n) - np.sin(lat_s))[:, None]
        * dlon_rad
    )

    return area


def finn_flux_to_tons_per_day(pm, lat, lon):
    """
    Convert mol/cm²/s → tons/day per grid cell
    (keeps time dimension!)
    """

    area = gridcell_area(lat, lon)

    # pm_g_m2_s = pm * CM2_TO_M2 * MOLAR_MASS
    pm_g_m2_s = (pm / AVOGADRO) * CM2_TO_M2 * MOLAR_MASS
    pm_g_m2_day = pm_g_m2_s * 86400

    pm_total_g = pm_g_m2_day * area
    pm_total_tons = pm_total_g * G_TO_TONS

    return pm_total_tons