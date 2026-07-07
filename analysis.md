```python
import sys
import os

project_root = "/glade/u/home/spatrin"
FIG_DIR = f"{project_root}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

print("Project root:", project_root)
```

    Project root: /glade/u/home/spatrin



```python
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns

import cartopy.crs as ccrs

from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from data_loading import load_epa_data, filter_june
from pipeline import load_finn_june_processed, run_daily_spatiotemporal_analysis
from utils import cumulative_curve

from spatial_utils import (
    epa_to_finn_grid,
    spatial_correlation,
    compute_overlap_metrics,
    basemap,
    classify_missed_fires
)

from geo_utils import get_georgia_mask, apply_ga_mask
```


```python
# Load EPA
epa = load_epa_data()
epa_june = filter_june(epa)

print("EPA June fires:", len(epa_june))
```

    EPA June fires: 2771



```python
# load FINN
finn = load_finn_june_processed()

lat = finn["lat"].values
lon = finn["lon"].values
print(finn["pm_tons"].shape)
```

    (30, 1799, 3600)


    /glade/u/home/spatrin/src/data_loading.py:42: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.
      ds = xr.open_dataset(



```python
finn_day = finn["pm_tons"].isel(time=0)
print(finn_day.shape)
```

    (1799, 3600)



```python
print(np.nanmax(finn["pm_tons"].values))
```

    1961.3107



```python
gdf = gpd.GeoDataFrame(
    epa_june,
    geometry=gpd.points_from_xy(epa_june["longitude"], epa_june["latitude"]),
    crs="EPSG:4326"
)
```


```python
epa_grid = epa_to_finn_grid(
    gdf,
    lat,
    lon,
    weight_type="pm25"   # can switch later
)

print("EPA grid shape:", epa_grid.shape)
```

    EPA grid shape: (1799, 3600)



```python
ga_mask = get_georgia_mask(lat, lon)

ga_pm = apply_ga_mask(
    finn["pm_tons"].values,
    ga_mask
)

finn_total_pm25 = np.nansum(ga_pm)

print(ga_mask.shape)
print(np.sum(ga_mask))
```

    (1799, 3600)
    1466



```python
def compute_miss_rate(metrics):
    return metrics["FN"] / (metrics["TP"] + metrics["FN"] + 1e-6)

def classify_all_days(gdf, finn, lat, lon):
    """Classify EPA fires as matched/missed using daily FINN fields."""
    results = []

    for date in sorted(gdf["date"].dt.normalize().unique()):
        epa_day = gdf[gdf["date"].dt.normalize() == date]

        if len(epa_day) == 0:
            continue

        finn_day = finn["pm_tons"].sel(time=date, method="nearest").values

        classified = classify_missed_fires(
            epa_day,
            finn_day,
            lat,
            lon
        )

        results.append(classified)

    return pd.concat(results, ignore_index=True)


def summarize_detection(df, group_col):
    summary = (
        df.groupby(group_col)["matched"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "detected"})
    )

    summary["miss_rate"] = 1 - (summary["detected"] / summary["count"])
    return summary


def emissions_summary(df):
    """Compute emissions + burned area breakdown."""
    total_pm = df["pm2.5"].sum()
    total_area = df["area"].sum()

    missed = df[df["matched"] == 0]
    detected = df[df["matched"] == 1]

    return {
        "total_pm25": total_pm,
        "detected_pm25": detected["pm2.5"].sum(),
        "missed_pm25": missed["pm2.5"].sum(),
        "total_area": total_area,
        "detected_area": detected["area"].sum(),
        "missed_area": missed["area"].sum()
    }
```

## RQ's


RQ1: How accurately does the FINN biomass burning emissions inventory represent
agricultural and prescribed fires in Georgia during June 2022 compared to EPA data?

This notebook evaluates:
1. Detection accuracy of FINN
2. Total emissions in FINN vs EPA
3. Fraction of emissions missed
4. Bias by fire type, siznumber
"""
``


```python
print("\n===== RQ1: FINN Detection Accuracy =====")

overall_metrics = run_daily_spatiotemporal_analysis(
    gdf,
    finn,
    lat,
    lon
)

overall_results = {
    "precision": overall_metrics["precision"],
    "recall": overall_metrics["recall"],
    "miss_rate": compute_miss_rate(overall_metrics),
    "TP": overall_metrics["TP"],
    "FN": overall_metrics["FN"],
    "FP": overall_metrics["FP"]
}

print("\nOverall metrics:")
print(overall_results)
```

    
    ===== RQ1: FINN Detection Accuracy =====
    
    Overall metrics:
    {'precision': np.float64(0.2678227355147924), 'recall': np.float64(0.07888762765102858), 'miss_rate': np.float64(0.9211123717814345), 'TP': np.int64(139), 'FN': np.int64(1623), 'FP': np.int64(380)}



```python
global_total = np.nansum(finn["pm_tons"].values)

ga_pm = apply_ga_mask(
    finn["pm_tons"].values,
    ga_mask
)

ga_total = np.nansum(ga_pm)

print("Global FINN:", global_total)
print("Georgia FINN:", ga_total)
```

    Global FINN: 3.3752022e+06
    Georgia FINN: 3767.5254



```python
epa_total_pm25 = gdf["pm2.5"].sum()
print("EPA total PM2.5:", epa_total_pm25)
print("FINN total PM2.5:", finn_total_pm25)
```

    EPA total PM2.5: 2674.6933349425713
    FINN total PM2.5: 3767.5254



```python
bias = finn_total_pm25 - epa_total_pm25
ratio = finn_total_pm25 / (epa_total_pm25 + 1e-6)

print("Difference (FINN - EPA):", bias)
print("Ratio (FINN/EPA):", ratio)
```

    Difference (FINN - EPA): 1092.8320556824287
    Ratio (FINN/EPA): 1.4085821877210873


FINN estimates approximately 41% more PM2.5 than EPA.


```python
finn_mean = np.nanmean(finn["pm_tons"].values, axis=0)

epa_classified = classify_missed_fires(
    gdf,
    finn_mean,
    lat,
    lon
)
epa_classified["area_km2"] = (
    epa_classified["area"] * 0.00404686
)
```


```python
summary = (
    epa_classified
    .groupby(["type", "matched"])
    .size()
    .unstack(fill_value=0)
)

summary.columns = ["missed", "matched"]

summary["miss_rate"] = summary["missed"] / (
    summary["missed"] + summary["matched"] + 1e-6
)

print(summary)
```

                  missed  matched  miss_rate
    type                                    
    Agricultural     599      390   0.605662
    Prescribed       563      485   0.537214
    Wildfire         469      265   0.638965


FINN misses wildfires the most, then Ag, then Prescribed


```python
daily_stats = []

for date in pd.to_datetime(gdf["date"]).dt.normalize().unique():

    epa_day = gdf[
        pd.to_datetime(gdf["date"]).dt.normalize() == date
    ]

    finn_day = finn["pm_tons"].sel(time=date, method="nearest")

    daily_stats.append({
        "date": date,
        "epa_fires": len(epa_day),
        "finn_active_cells": np.sum(finn_day.values > 1e-3)
    })

daily_df = pd.DataFrame(daily_stats)
print(daily_df)
```

             date  epa_fires  finn_active_cells
    0  2022-06-01        123              13901
    1  2022-06-02        142              13785
    2  2022-06-03        157              14415
    3  2022-06-04         76              14601
    4  2022-06-05         23              13977
    5  2022-06-06        111              14638
    6  2022-06-07        105              14753
    7  2022-06-08        113              15177
    8  2022-06-09         93              13888
    9  2022-06-10        142              14629
    10 2022-06-11         91              15097
    11 2022-06-12         38              13849
    12 2022-06-13        120              14655
    13 2022-06-14        100              15466
    14 2022-06-15         85              15333
    15 2022-06-16         86              15484
    16 2022-06-17         93              14743
    17 2022-06-18         70              13465
    18 2022-06-19         40              13082
    19 2022-06-20        102              12873
    20 2022-06-21        106              14702
    21 2022-06-22        124              14451
    22 2022-06-23        109              14599
    23 2022-06-24        101              15652
    24 2022-06-25         56              14245
    25 2022-06-26         44              13933
    26 2022-06-27         98              16493
    27 2022-06-28         91              18225
    28 2022-06-29         68              16534
    29 2022-06-30         64              17571



```python
print("\n===== Detection by Fire Type =====")

results_by_type = {}

for fire_type in gdf["type"].unique():
    subset = gdf[gdf["type"] == fire_type]

    metrics = run_daily_spatiotemporal_analysis(
        subset,
        finn,
        lat,
        lon
    )

    results_by_type[fire_type] = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "miss_rate": compute_miss_rate(metrics),
        "TP": metrics["TP"],
        "FN": metrics["FN"]
    }

print("\nFire-type metrics:")
for k, v in results_by_type.items():
    print(f"{k}: {v}")
```

    
    ===== Detection by Fire Type =====
    
    Fire-type metrics:
    Agricultural: {'precision': np.float64(0.12909441208266972), 'recall': np.float64(0.07322404363582072), 'miss_rate': np.float64(0.9267759552712831), 'TP': np.int64(67), 'FN': np.int64(848)}
    Wildfire: {'precision': np.float64(0.05202312128704601), 'recall': np.float64(0.07297297277574873), 'miss_rate': np.float64(0.9270270245215486), 'TP': np.int64(27), 'FN': np.int64(343)}
    Prescribed: {'precision': np.float64(0.1098265893837638), 'recall': np.float64(0.10857142836462585), 'miss_rate': np.float64(0.8914285697306122), 'TP': np.int64(57), 'FN': np.int64(468)}



```python
print("\n===== Classifying Fires =====")

epa_match_df = classify_all_days(gdf, finn, lat, lon)

print("Total classified fires:", len(epa_match_df))
print("Matched fires:", epa_match_df["matched"].sum())
print("Missed fires:", (epa_match_df["matched"] == 0).sum())
```

    
    ===== Classifying Fires =====
    Total classified fires: 2771
    Matched fires: 260
    Missed fires: 2511



```python
epa_match_df["area_km2"] = (
    epa_match_df["area"] * 0.00404686
)
```


```python
print("\n===== Missed Emissions by Fire Type =====")

missed = epa_match_df[epa_match_df["matched"] == 0]

missed_by_type = missed.groupby("type").agg({
    "pm2.5": "sum",
    "area": "sum"
})

print(missed_by_type)
```

    
    ===== Missed Emissions by Fire Type =====
                       pm2.5          area
    type                                  
    Agricultural  320.914838  23852.500000
    Prescribed    784.146780  23538.276000
    Wildfire      704.349086   8988.668418



```python
print("\n===== Emissions & Burned Area Accounting =====")

summary_totals = emissions_summary(epa_match_df)

for k, v in summary_totals.items():
    print(f"{k}: {v}")
```

    
    ===== Emissions & Burned Area Accounting =====
    total_pm25: 2674.6933349425713
    detected_pm25: 865.2826309549013
    missed_pm25: 1809.4107039876699
    total_area: 78836.479473865
    detected_area: 22457.035056068
    missed_area: 56379.444417797



```python
pm_bins = [0, 0.01, 0.1, 1, 10, np.inf]

pm_labels = [
    "<0.01",
    "0.01–0.1",
    "0.1–1",
    "1–10",
    ">10"
]

epa_match_df["pm_bin"] = pd.cut(
    epa_match_df["pm2.5"],
    bins=pm_bins,
    labels=pm_labels,
    include_lowest=True
)

pm_size_results = summarize_detection(epa_match_df, "pm_bin")

print("\nPM2.5 Size Results:")
print(pm_size_results)
```

    
    PM2.5 Size Results:
              detected  count  miss_rate
    pm_bin                              
    <0.01           27    506   0.946640
    0.01–0.1        84   1066   0.921201
    0.1–1           92    874   0.894737
    1–10            47    290   0.837931
    >10             10     35   0.714286


    /glade/derecho/scratch/spatrin/tmp/ipykernel_91762/2527162575.py:30: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby(group_col)["matched"]



```python
print(
    epa_match_df["pm_bin"]
    .value_counts()
    .sort_index()
)
```

    pm_bin
    <0.01        506
    0.01–0.1    1066
    0.1–1        874
    1–10         290
    >10           35
    Name: count, dtype: int64



```python
area_bins = [0, 0.01, 0.1, 1, 10, np.inf]

area_labels = [
    "<0.01",
    "0.01–0.1",
    "0.1–1",
    "1–10",
    ">10"
]

epa_match_df["area_bin"] = pd.cut(
    epa_match_df["area_km2"],
    bins=area_bins,
    labels=area_labels,
    include_lowest=True
)

area_size_results = summarize_detection(epa_match_df, "area_bin")

print("\nBurned Area Results:")
print(area_size_results)
```

    
    Burned Area Results:
              detected  count  miss_rate
    area_bin                            
    <0.01           71   1289   0.944919
    0.01–0.1        83    848   0.902123
    0.1–1           90    592   0.847973
    1–10            15     41   0.634146
    >10              1      1   0.000000


    /glade/derecho/scratch/spatrin/tmp/ipykernel_91762/2527162575.py:30: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby(group_col)["matched"]



```python
print("\n===== Size Dependence by Fire Type =====")

combo_results = (
    epa_match_df
    .groupby(["type", "pm_bin"], observed=True)["matched"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "detected"})
)

combo_results["miss_rate"] = 1 - (combo_results["detected"] / combo_results["count"])

print(combo_results)
```

    
    ===== Size Dependence by Fire Type =====
                           detected  count  miss_rate
    type         pm_bin                              
    Agricultural <0.01            2     85   0.976471
                 0.01–0.1        27    421   0.935867
                 0.1–1           33    373   0.911528
                 1–10            16    110   0.854545
    Prescribed   <0.01           19    195   0.902564
                 0.01–0.1        39    379   0.897098
                 0.1–1           42    318   0.867925
                 1–10            23    133   0.827068
                 >10              8     23   0.652174
    Wildfire     <0.01            6    226   0.973451
                 0.01–0.1        18    266   0.932331
                 0.1–1           17    183   0.907104
                 1–10             8     47   0.829787
                 >10              2     12   0.833333



```python
print("\n===== Matched vs Missed Comparison =====")

matched = epa_match_df[epa_match_df["matched"] == 1]
missed = epa_match_df[epa_match_df["matched"] == 0]
detection_rate = len(matched)/2771 * 100

print("\nCounts:")
print("Matched:", len(matched))
print("Missed:", len(missed))
print("Detection (%):", detection_rate)
```

    
    ===== Matched vs Missed Comparison =====
    
    Counts:
    Matched: 260
    Missed: 2511
    Detection (%): 9.382894261999278



```python
total_by_type = epa_match_df.groupby("type").agg({
    "pm2.5": "sum",
    "area": "sum"
})

fraction_missed_by_type = missed_by_type / (total_by_type + 1e-6)

print("\nFraction missed by fire type:")
print(fraction_missed_by_type)
```

    
    Fraction missed by fire type:
                     pm2.5      area
    type                            
    Agricultural  0.869687  0.860868
    Prescribed    0.505592  0.569796
    Wildfire      0.933228  0.915435



```python
print("\n===== Detection Probability by PM2.5 Size =====")

# pm_detection = epa_match_df.groupby(pm_bin, observed=True)["matched"].mean()
pm_detection = (
    epa_match_df
    .groupby("pm_bin", observed=True)
    .agg(
        detection_probability=("matched", "mean"),
        count=("matched", "size")
    )
)

print(pm_detection)
```

    
    ===== Detection Probability by PM2.5 Size =====
              detection_probability  count
    pm_bin                                
    <0.01                  0.053360    506
    0.01–0.1               0.078799   1066
    0.1–1                  0.105263    874
    1–10                   0.162069    290
    >10                    0.285714     35


Detection probability increases monotonically with emissions, suggesting FINN preferentially captures larger, more emissive fires.


```python
print("\n===== Detection Probability by Burned Area =====")

area_detection = (
    epa_match_df
    .groupby("area_bin", observed=True)
    .agg(
        detection_probability=("matched", "mean"),
        count=("matched", "size")
    )
)
print(area_detection)
```

    
    ===== Detection Probability by Burned Area =====
              detection_probability  count
    area_bin                              
    <0.01                  0.055081   1289
    0.01–0.1               0.097877    848
    0.1–1                  0.152027    592
    1–10                   0.365854     41
    >10                    1.000000      1


Detection probability increases strongly with fire size, but remains below 50% even for the largest fires.


```python
print("\n===== Missed vs Detected Totals =====")

total_pm = epa_match_df["pm2.5"].sum()
missed_pm = missed["pm2.5"].sum()

total_area = epa_match_df["area"].sum()
missed_area = missed["area"].sum()

print("Total PM2.5:", total_pm)
print("Missed PM2.5:", missed_pm)
print("Fraction missed PM2.5:", missed_pm / (total_pm + 1e-6))

print("\nTotal Burned Area:", total_area)
print("Missed Burned Area:", missed_area)
print("Fraction missed area:", missed_area / (total_area + 1e-6))

print("\nAverage PM2.5:")
print("Matched mean:", matched["pm2.5"].mean())
print("Missed mean:", missed["pm2.5"].mean())

print("\nAverage Burned Area:")
print("Matched mean:", matched["area"].mean())
print("Missed mean:", missed["area"].mean())
```

    
    ===== Missed vs Detected Totals =====
    Total PM2.5: 2674.6933349425713
    Missed PM2.5: 1809.4107039876699
    Fraction missed PM2.5: 0.6764927700954649
    
    Total Burned Area: 78836.479473865
    Missed Burned Area: 56379.444417797
    Fraction missed area: 0.7151441159390198
    
    Average PM2.5:
    Matched mean: 3.3280101190573124
    Missed mean: 0.7205936694494902
    
    Average Burned Area:
    Matched mean: 86.3732117541077
    Missed mean: 22.452984634726004



```python
print("\n===== Missed Emissions by Fire Type =====")

missed_by_type = missed.groupby("type").agg({
    "pm2.5": "sum",
    "area": "sum"
})

print(missed_by_type)
```

    
    ===== Missed Emissions by Fire Type =====
                       pm2.5          area
    type                                  
    Agricultural  320.914838  23852.500000
    Prescribed    784.146780  23538.276000
    Wildfire      704.349086   8988.668418



```python
daily_results = run_daily_spatiotemporal_analysis(
    gdf,
    finn,
    lat,
    lon
)

daily_confusion = daily_results["daily"]
daily_confusion.head()
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
      <th>date</th>
      <th>TP</th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>miss_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-01</td>
      <td>7</td>
      <td>76</td>
      <td>22</td>
      <td>NaN</td>
      <td>0.915663</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-06-02</td>
      <td>10</td>
      <td>85</td>
      <td>15</td>
      <td>NaN</td>
      <td>0.894737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-06-03</td>
      <td>14</td>
      <td>85</td>
      <td>7</td>
      <td>NaN</td>
      <td>0.858586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-06-04</td>
      <td>4</td>
      <td>49</td>
      <td>13</td>
      <td>NaN</td>
      <td>0.924528</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-06-05</td>
      <td>0</td>
      <td>20</td>
      <td>11</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
monthly_totals = daily_confusion[
    ["TP", "FN", "FP"]
].sum()

monthly_totals
```




    TP     139
    FN    1623
    FP     380
    dtype: int64




```python
daily_confusion["detection_rate"] = (
    daily_confusion["TP"]
    /
    (
        daily_confusion["TP"]
        + daily_confusion["FN"]
    )
)
```


```python
daily_merged = daily_confusion.merge(
    daily_df,
    on="date"
)
```


```python
daily_merged.corr(numeric_only=True)
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
      <th>TN</th>
      <th>miss_rate</th>
      <th>detection_rate</th>
      <th>epa_fires</th>
      <th>finn_active_cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TP</th>
      <td>1.000000</td>
      <td>0.648078</td>
      <td>0.416590</td>
      <td>NaN</td>
      <td>-0.868196</td>
      <td>0.868196</td>
      <td>0.747323</td>
      <td>-0.076080</td>
    </tr>
    <tr>
      <th>FN</th>
      <td>0.648078</td>
      <td>1.000000</td>
      <td>0.140711</td>
      <td>NaN</td>
      <td>-0.316805</td>
      <td>0.316805</td>
      <td>0.971642</td>
      <td>0.076131</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>0.416590</td>
      <td>0.140711</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.424454</td>
      <td>0.424454</td>
      <td>0.230833</td>
      <td>-0.253156</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>miss_rate</th>
      <td>-0.868196</td>
      <td>-0.316805</td>
      <td>-0.424454</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>-0.448674</td>
      <td>0.002019</td>
    </tr>
    <tr>
      <th>detection_rate</th>
      <td>0.868196</td>
      <td>0.316805</td>
      <td>0.424454</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>0.448674</td>
      <td>-0.002019</td>
    </tr>
    <tr>
      <th>epa_fires</th>
      <td>0.747323</td>
      <td>0.971642</td>
      <td>0.230833</td>
      <td>NaN</td>
      <td>-0.448674</td>
      <td>0.448674</td>
      <td>1.000000</td>
      <td>0.032016</td>
    </tr>
    <tr>
      <th>finn_active_cells</th>
      <td>-0.076080</td>
      <td>0.076131</td>
      <td>-0.253156</td>
      <td>NaN</td>
      <td>0.002019</td>
      <td>-0.002019</td>
      <td>0.032016</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary_table = (
    epa_match_df
    .groupby("type")
    .agg(
        n_fires=("type","size"),
        total_pm25=("pm2.5","sum"),
        median_pm25=("pm2.5","median"),
        total_area_km2=("area_km2","sum"),
        median_area_km2=("area_km2","median")
    )
)

summary_table
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
      <th>n_fires</th>
      <th>total_pm25</th>
      <th>median_pm25</th>
      <th>total_area_km2</th>
      <th>median_area_km2</th>
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
      <td>989</td>
      <td>369.000556</td>
      <td>0.093786</td>
      <td>112.128373</td>
      <td>0.024281</td>
    </tr>
    <tr>
      <th>Prescribed</th>
      <td>1048</td>
      <td>1550.947821</td>
      <td>0.073757</td>
      <td>167.175669</td>
      <td>0.020234</td>
    </tr>
    <tr>
      <th>Wildfire</th>
      <td>734</td>
      <td>754.744957</td>
      <td>0.038688</td>
      <td>39.736153</td>
      <td>0.005241</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    np.log10(
        epa_classified["pm2.5"] + 0.001
    ),
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("log10(PM₂.₅)")
ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
```


    
![png](output_44_0.png)
    



```python
fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    np.log10(
        epa_classified["area_km2"] + 1e-4
    ),
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("log10(Burned Area km²)")
ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
```


    
![png](output_45_0.png)
    


## stat tests


```python
print("\n===== Chi-Square: Fire Type vs Detection =====")

contingency_type = pd.crosstab(
    epa_match_df["type"],
    epa_match_df["matched"]
)

chi2_type, p_type, _, _ = chi2_contingency(contingency_type)

print("Chi2:", chi2_type)
print("p-value:", p_type)
```

    
    ===== Chi-Square: Fire Type vs Detection =====
    Chi2: 19.697013582370474
    p-value: 5.28260144228432e-05



```python
print("\n===== Chi-Square: Size vs Detection =====")

contingency_size = pd.crosstab(
    epa_match_df["pm_bin"],
    epa_match_df["matched"]
)

chi2_size, p_size, _, _ = chi2_contingency(contingency_size)

print("Chi2:", chi2_size)
print("p-value:", p_size)
```

    
    ===== Chi-Square: Size vs Detection =====
    Chi2: 44.96224476024047
    p-value: 4.048474940016092e-09



```python
print("\n===== Logistic Regression =====")

df_model = epa_match_df.copy()

df_model["type_encoded"] = df_model["type"].map({
    "Agricultural": 0,
    "Prescribed": 1,
    "Wildfire": 2
})

df_model["pm_numeric"] = df_model["pm2.5"].fillna(0)
df_model["area_numeric"] = df_model["area"].fillna(0)

X = df_model[["type_encoded", "pm_numeric", "area_numeric"]]
X = sm.add_constant(X)

y = df_model["matched"]

model = sm.Logit(y, X).fit(disp=0)

print(model.summary())
```

    
    ===== Logistic Regression =====
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                matched   No. Observations:                 2771
    Model:                          Logit   Df Residuals:                     2767
    Method:                           MLE   Df Model:                            3
    Date:                Mon, 06 Jul 2026   Pseudo R-squ.:                 0.02522
    Time:                        09:40:14   Log-Likelihood:                -840.88
    converged:                       True   LL-Null:                       -862.64
    Covariance Type:            nonrobust   LLR p-value:                 1.914e-09
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const           -2.3889      0.104    -22.918      0.000      -2.593      -2.185
    type_encoded     0.0036      0.085      0.042      0.966      -0.164       0.171
    pm_numeric      -0.0193      0.009     -2.110      0.035      -0.037      -0.001
    area_numeric     0.0037      0.001      4.885      0.000       0.002       0.005
    ================================================================================



```python
print("\n===== Emissions Agreement =====")

# EPA daily totals
epa_daily = gdf.groupby(gdf["date"].dt.normalize())["pm2.5"].sum()

# FINN daily totals
finn_daily = finn["pm_tons"].sum(dim=["lat", "lon"]).to_pandas()

# Align
combined = pd.concat([epa_daily, finn_daily], axis=1).dropna()
combined.columns = ["EPA", "FINN"]

bias = np.mean(combined["FINN"] - combined["EPA"])
rmse = np.sqrt(np.mean((combined["FINN"] - combined["EPA"])**2))
corr = combined.corr().iloc[0, 1]

print("Bias (FINN - EPA):", bias)
print("RMSE:", rmse)
print("Correlation:", corr)
```

    
    ===== Emissions Agreement =====
    Bias (FINN - EPA): 112417.58912841855
    RMSE: 118133.68573458579
    Correlation: 0.0060665285319883916



```python
print("\n===== Fractional Bias =====")

fb = np.mean(
    (combined["FINN"] - combined["EPA"]) /
    (combined["FINN"] + combined["EPA"] + 1e-6)
)

print("Fractional Bias:", fb)
```

    
    ===== Fractional Bias =====
    Fractional Bias: 0.9983123141158923



```python
print("\n--- PM2.5 Comparison ---")

t_pm, p_pm = ttest_ind(
    matched["pm2.5"],
    missed["pm2.5"],
    nan_policy="omit"
)

ks_pm, ks_p_pm = ks_2samp(
    matched["pm2.5"],
    missed["pm2.5"]
)

print("t-test statistic:", t_pm)
print("t-test p-value:", p_pm)

print("KS statistic:", ks_pm)
print("KS p-value:", ks_p_pm)
```

    
    --- PM2.5 Comparison ---
    t-test statistic: 4.70579831730412
    t-test p-value: 2.652968280436012e-06
    KS statistic: 0.19808228410378947
    KS p-value: 1.4170329736702622e-08



```python
print("\n--- Burned Area Comparison ---")

t_area, p_area = ttest_ind(
    matched["area"],
    missed["area"],
    nan_policy="omit"
)

ks_area, ks_p_area = ks_2samp(
    matched["area"],
    missed["area"]
)

print("t-test statistic:", t_area)
print("t-test p-value:", p_area)

print("KS statistic:", ks_area)
print("KS p-value:", ks_p_area)
```

    
    --- Burned Area Comparison ---
    t-test statistic: 8.501435105039706
    t-test p-value: 3.015300806980007e-17
    KS statistic: 0.240432864626413
    KS p-value: 1.8333980806623497e-12


## Plots


```python
sns.set_theme(style="white", font_scale=1.1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False
})
```


```python
type_summary = (
    epa_match_df
    .groupby("type")["matched"]
    .agg(["mean","count","std"])
    .reset_index()
)

type_summary["se"] = (
    type_summary["std"] /
    np.sqrt(type_summary["count"])
)
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(5.5, 4))

bars = ax.bar(
    type_summary["type"],
    type_summary["mean"],
    yerr=1.96 * type_summary["se"],
    color="0.4",
    edgecolor="black",
    linewidth=1.0,
    capsize=4
)

# percentage labels
for bar, n in zip(bars, type_summary["count"]):

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.025,
        f"{height:.1%}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        bar.get_x() + bar.get_width()/2,
        0.005,
        f"n={n}",
        ha="center",
        fontsize=9,
        color="white"
    )

ax.set_title(
    "Detection Probability by Fire Type",
    fontsize=14,
    pad=10
)

ax.set_ylabel("Detection Probability (%)")
ax.set_xlabel("")

ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_ylim(0, 0.16)

sns.despine()

plt.tight_layout()
plt.show()
```


    
![png](output_56_0.png)
    


Figure X. Detection probability of EPA-reported fires by fire type. Error bars represent 95% confidence intervals. Detection probability was highest for prescribed fires (12.5%), followed by agricultural fires (7.9%) and wildfires (6.9%).


```python
epa_match_df.groupby("pm_bin")["pm2.5"].agg(
    ["count", "min", "max", "median"]
)
```

    /glade/derecho/scratch/spatrin/tmp/ipykernel_91762/1215404032.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      epa_match_df.groupby("pm_bin")["pm2.5"].agg(





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
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>median</th>
    </tr>
    <tr>
      <th>pm_bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;0.01</th>
      <td>506</td>
      <td>0.000001</td>
      <td>0.009981</td>
      <td>0.004021</td>
    </tr>
    <tr>
      <th>0.01–0.1</th>
      <td>1066</td>
      <td>0.010004</td>
      <td>0.099985</td>
      <td>0.035076</td>
    </tr>
    <tr>
      <th>0.1–1</th>
      <td>874</td>
      <td>0.100066</td>
      <td>0.996629</td>
      <td>0.321693</td>
    </tr>
    <tr>
      <th>1–10</th>
      <td>290</td>
      <td>1.008306</td>
      <td>9.719234</td>
      <td>1.676409</td>
    </tr>
    <tr>
      <th>&gt;10</th>
      <td>35</td>
      <td>10.370189</td>
      <td>280.913234</td>
      <td>16.357494</td>
    </tr>
  </tbody>
</table>
</div>




```python
pm_size_results = (
    epa_match_df
    .groupby("pm_bin", observed=True)["matched"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6,4))

sns.pointplot(
    data=pm_size_results,
    x="pm_bin",
    y="matched",
    color="black"
)

plt.ylabel("Detection Probability")
plt.xlabel("PM$_{2.5}$ Emissions (t day$^{-1}$)")
plt.title("Detection Probability Increases With Fire Emissions")
plt.ylim(0,0.5)
sns.despine()
plt.tight_layout()
plt.show()
```


    
![png](output_59_0.png)
    



```python
area_size_results = (
    epa_match_df
    .groupby('area_bin')['matched']
    .mean()
    .reset_index()
)

plt.figure(figsize=(10,4))
sns.barplot(data=area_size_results, x='area_bin', y='matched', color='black')

plt.xlabel('Burned Area Bin')
plt.ylabel('Detection Probability')
plt.title('Detection vs Burned Area')

plt.ylim(0, 0.6)
sns.despine()
plt.tight_layout()
plt.show()
```

    /glade/derecho/scratch/spatrin/tmp/ipykernel_91762/308943031.py:3: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      .groupby('area_bin')['matched']



    
![png](output_60_1.png)
    



```python
print(daily_stats.columns)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[54], line 1
    ----> 1 print(daily_stats.columns)


    AttributeError: 'list' object has no attribute 'columns'



```python
daily_stats = pd.DataFrame(daily_stats)
plt.figure(figsize=(7,4))

plt.plot(daily_stats['date'], daily_stats['epa_fires'], label='epa_fires')
plt.plot(daily_stats['date'], daily_stats['finn_active_cells'], label='finn_active_cells')

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Daily Fires: EPA vs FINN')
plt.xticks(rotation=45)
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()
```


    
![png](output_62_0.png)
    



```python
fig, ax1 = plt.subplots(figsize=(8,4))

# EPA
ax1.plot(
    daily_stats['date'],
    daily_stats['epa_fires'],
    color='black',
    linewidth=2,
    label='EPA fires'
)

ax1.set_ylabel('EPA fires per day', color='black')
ax1.tick_params(axis='y', colors='black')

# FINN
ax2 = ax1.twinx()

ax2.plot(
    daily_stats['date'],
    daily_stats['finn_active_cells'],
    color='firebrick',
    linewidth=2,
    label='FINN active cells'
)

ax2.set_ylabel('FINN active cells per day', color='firebrick')
ax2.tick_params(axis='y', colors='firebrick')

plt.title('Daily Fire Activity: EPA vs FINN')
ax1.set_xlabel('Date')

fig.autofmt_xdate()
sns.despine()

plt.tight_layout()
plt.show()
```


    
![png](output_63_0.png)
    



```python
epa_match_df["log_pm25"] = np.log10(
    epa_match_df["pm2.5"] + 0.01
)

plt.figure(figsize=(5,4))

sns.kdeplot(
    data=epa_match_df,
    x="log_pm25",
    hue="matched",
    fill=True,
    common_norm=False,
    alpha=0.4
)

plt.xlabel("log10(PM2.5)")
plt.ylabel("Density")

sns.despine()
plt.tight_layout()
plt.show()
```


    
![png](output_64_0.png)
    



```python
epa_match_df["log_area"] = np.log10(
    epa_match_df["area"] + 1
)

plt.figure(figsize=(5,4))

sns.kdeplot(
    data=epa_match_df,
    x="log_area",
    hue="matched",
    fill=True,
    common_norm=False,
    alpha=0.4
)

plt.xlabel("log10(Burned Area)")
plt.ylabel("Density")

sns.despine()
plt.tight_layout()
plt.show()
```


    
![png](output_65_0.png)
    



```python
violin_color = "#9ecae1"   # light blue
point_color  = "#2171b5"   # darker blue

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 4.8),
)

# Panel A: PM2.5

sns.violinplot(
    data=epa_match_df,
    x="Detection Status",
    y="log_pm25",
    inner=None,
    color=violin_color,
    saturation=1,
    cut=0,
    linewidth=1.2,
    ax=axes[0]
)

sns.boxplot(
    data=epa_match_df,
    x="Detection Status",
    y="log_pm25",
    width=0.18,
    showcaps=True,
    fliersize=0,
    boxprops=dict(
        facecolor="white",
        edgecolor="black",
        linewidth=1.2
    ),
    medianprops=dict(
        color="black",
        linewidth=2
    ),
    whiskerprops=dict(
        color="black",
        linewidth=1.2
    ),
    capprops=dict(
        color="black",
        linewidth=1.2
    ),
    ax=axes[0]
)

sns.stripplot(
    data=epa_match_df.sample(
        min(800, len(epa_match_df)),
        random_state=42
    ),
    x="Detection Status",
    y="log_pm25",
    color=point_color,
    alpha=0.35,
    size=2.8,
    jitter=0.22,
    ax=axes[0]
)

axes[0].set_title(
    r"PM$_{2.5}$ Emissions",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_ylabel(
    r'log$_{10}$(PM$_{2.5}$ emissions [tons day$^{-1}$])'
)

axes[0].set_xlabel("")

axes[0].text(
    0.02, 0.98,
    "A",
    transform=axes[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

axes[0].text(
    0.5,
    0.95,
    r"$p < 0.001$",
    transform=axes[0].transAxes,
    ha="center",
    fontsize=12
)

# Panel B: Area

sns.violinplot(
    data=epa_match_df,
    x="Detection Status",
    y="log_area",
    inner=None,
    color=violin_color,
    saturation=1,
    cut=0,
    linewidth=1.2,
    ax=axes[1]
)

sns.boxplot(
    data=epa_match_df,
    x="Detection Status",
    y="log_area",
    width=0.18,
    showcaps=True,
    fliersize=0,
    boxprops=dict(
        facecolor="white",
        edgecolor="black",
        linewidth=1.2
    ),
    medianprops=dict(
        color="black",
        linewidth=2
    ),
    whiskerprops=dict(
        color="black",
        linewidth=1.2
    ),
    capprops=dict(
        color="black",
        linewidth=1.2
    ),
    ax=axes[1]
)

sns.stripplot(
    data=epa_match_df.sample(
        min(800, len(epa_match_df)),
        random_state=42
    ),
    x="Detection Status",
    y="log_area",
    color=point_color,
    alpha=0.35,
    size=2.8,
    jitter=0.22,
    ax=axes[1]
)

axes[1].set_title(
    "Burned Area",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_ylabel(
    r'log$_{10}$(Burned area [acres])'
)

axes[1].set_xlabel("")

axes[1].text(
    0.02, 0.98,
    "B",
    transform=axes[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top"
)

axes[1].text(
    0.5,
    0.95,
    r"$p < 0.001$",
    transform=axes[1].transAxes,
    ha="center",
    fontsize=12
)

n_missed = (epa_match_df["Detection Status"] == "Missed").sum()
n_detected = (epa_match_df["Detection Status"] == "Detected").sum()

for ax in axes:

    ymin, ymax = ax.get_ylim()

    # lowered from current position
    y_loc = ymin + 0.015 * (ymax - ymin)

    ax.text(
        0,
        y_loc - 0.1,
        f"n = {n_missed:,}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black"
    )

    ax.text(
        1,
        y_loc-0.05,
        f"n = {n_detected:,}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black"
    )

    sns.despine(ax=ax)

fig.suptitle(
    "Fire Size and Detection Status",
    fontsize=15,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()
plt.show()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[59], line 11
          4 fig, axes = plt.subplots(
          5     1, 2,
          6     figsize=(10, 4.8),
          7 )
          9 # Panel A: PM2.5
    ---> 11 sns.violinplot(
         12     data=epa_match_df,
         13     x="Detection Status",
         14     y="log_pm25",
         15     inner=None,
         16     color=violin_color,
         17     saturation=1,
         18     cut=0,
         19     linewidth=1.2,
         20     ax=axes[0]
         21 )
         23 sns.boxplot(
         24     data=epa_match_df,
         25     x="Detection Status",
       (...)     47     ax=axes[0]
         48 )
         50 sns.stripplot(
         51     data=epa_match_df.sample(
         52         min(800, len(epa_match_df)),
       (...)     61     ax=axes[0]
         62 )


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/categorical.py:1725, in violinplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, fill, inner, split, width, dodge, gap, linewidth, linecolor, cut, gridsize, bw_method, bw_adjust, density_norm, common_norm, hue_norm, formatter, log_scale, native_scale, legend, scale, scale_hue, bw, inner_kws, ax, **kwargs)
       1714 def violinplot(
       1715     data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
       1716     orient=None, color=None, palette=None, saturation=.75, fill=True,
       (...)   1722     inner_kws=None, ax=None, **kwargs,
       1723 ):
    -> 1725     p = _CategoricalPlotter(
       1726         data=data,
       1727         variables=dict(x=x, y=y, hue=hue),
       1728         order=order,
       1729         orient=orient,
       1730         color=color,
       1731         legend=legend,
       1732     )
       1734     if ax is None:
       1735         ax = plt.gca()


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/categorical.py:67, in _CategoricalPlotter.__init__(self, data, variables, order, orient, require_numeric, color, legend)
         56 def __init__(
         57     self,
         58     data=None,
       (...)     64     legend="auto",
         65 ):
    ---> 67     super().__init__(data=data, variables=variables)
         69     # This method takes care of some bookkeeping that is necessary because the
         70     # original categorical plots (prior to the 2021 refactor) had some rules that
         71     # don't fit exactly into VectorPlotter logic. It may be wise to have a second
       (...)     76     # default VectorPlotter rules. If we do decide to make orient part of the
         77     # _base variable assignment, we'll want to figure out how to express that.
         78     if self.input_format == "wide" and orient in ["h", "y"]:


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/_base.py:634, in VectorPlotter.__init__(self, data, variables)
        629 # var_ordered is relevant only for categorical axis variables, and may
        630 # be better handled by an internal axis information object that tracks
        631 # such information and is set up by the scale_* methods. The analogous
        632 # information for numeric axes would be information about log scales.
        633 self._var_ordered = {"x": False, "y": False}  # alt., used DefaultDict
    --> 634 self.assign_variables(data, variables)
        636 # TODO Lots of tests assume that these are called to initialize the
        637 # mappings to default values on class initialization. I'd prefer to
        638 # move away from that and only have a mapping when explicitly called.
        639 for var in ["hue", "size", "style"]:


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/_base.py:679, in VectorPlotter.assign_variables(self, data, variables)
        674 else:
        675     # When dealing with long-form input, use the newer PlotData
        676     # object (internal but introduced for the objects interface)
        677     # to centralize / standardize data consumption logic.
        678     self.input_format = "long"
    --> 679     plot_data = PlotData(data, variables)
        680     frame = plot_data.frame
        681     names = plot_data.names


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/_core/data.py:58, in PlotData.__init__(self, data, variables)
         51 def __init__(
         52     self,
         53     data: DataSource,
         54     variables: dict[str, VariableSpec],
         55 ):
         57     data = handle_data_source(data)
    ---> 58     frame, names, ids = self._assign_variables(data, variables)
         60     self.frame = frame
         61     self.names = names


    File /glade/u/apps/opt/miniforge/envs/npl-2026a/lib/python3.13/site-packages/seaborn/_core/data.py:232, in PlotData._assign_variables(self, data, variables)
        230     else:
        231         err += "An entry with this name does not appear in `data`."
    --> 232     raise ValueError(err)
        234 else:
        235 
        236     # Otherwise, assume the value somehow represents data
        237 
        238     # Ignore empty data structures
        239     if isinstance(val, Sized) and len(val) == 0:


    ValueError: Could not interpret value `Detection Status` for `x`. An entry with this name does not appear in `data`.



    
![png](output_66_1.png)
    


Detected fires are systematically larger than missed fires in both PM₂.₅ emissions and burned area.


```python
plt.figure(figsize=(9,4))

plt.plot(
    daily_confusion["date"],
    daily_confusion["TP"],
    label="True Positives",
    color="forestgreen",
    linewidth=2
)

plt.plot(
    daily_confusion["date"],
    daily_confusion["FN"],
    label="False Negatives",
    color="firebrick",
    linewidth=2
)

plt.ylabel("Count")
plt.xlabel("Date")
plt.title("Daily Detection Performance")

plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](output_68_0.png)
    



```python
daily_confusion["Detected"] = daily_confusion["TP"]
daily_confusion["Missed"] = daily_confusion["FN"]

fig, ax = plt.subplots(figsize=(10,4))

ax.bar(
    daily_confusion["date"],
    daily_confusion["Detected"],
    color="#3182bd",
    label="Detected"
)

ax.bar(
    daily_confusion["date"],
    daily_confusion["Missed"],
    bottom=daily_confusion["Detected"],
    color="#d9d9d9",
    label="Missed"
)

ax.set_ylabel("EPA Fires")

ax.set_title(
    "Daily EPA Fires Detected and Missed by FINN"
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
```


    
![png](output_69_0.png)
    



```python
daily_confusion["daily_fires"] = (
    daily_confusion["TP"] +
    daily_confusion["FN"]
)

fig, ax = plt.subplots(figsize=(10,4))

ax.plot(
    daily_confusion["date"],
    daily_confusion["daily_fires"],
    color="black",
    linewidth=2,
    label="EPA Fires"
)

ax.plot(
    daily_confusion["date"],
    daily_confusion["TP"],
    color="#2ca25f",
    linewidth=2,
    label="Detected by FINN"
)

ax.set_ylabel("Fire Count")

ax.legend()

ax.set_title(
    "Daily EPA Fires and FINN Detections"
)

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
```


    
![png](output_70_0.png)
    



```python
plt.figure(figsize=(9,4))

plt.plot(
    daily_confusion["date"],
    daily_confusion["detection_rate"],
    marker="o",
    linewidth=2
)

plt.ylabel("Detection Rate")
plt.xlabel("Date")

plt.title(
    "Daily Detection Rate of EPA Fires by FINN"
)

plt.ylim(0, 1)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](output_71_0.png)
    



```python
r, p = pearsonr(
    daily_merged["epa_fires"],
    daily_merged["detection_rate"]
)

sns.regplot(
    data=daily_merged,
    x="epa_fires",
    y="detection_rate",
    scatter_kws={"s":60},
    line_kws={"color":"firebrick"}
)

plt.title(
    f"Detection Rate vs Fire Activity\nr = {r:.2f}, p = {p:.3f}"
)

plt.xlabel("EPA Fires per Day")
plt.ylabel("Detection Rate")

plt.tight_layout()
plt.show()
```


    
![png](output_72_0.png)
    



```python
fig, ax1 = plt.subplots(figsize=(10,4))

ax1.plot(
    daily_merged["date"],
    daily_merged["detection_rate"],
    color="#2171b5",
    marker="o",
    linewidth=2,
)

ax1.set_ylabel(
    "Detection Rate",
    color="#2171b5"
)

ax1.set_ylim(0,0.25)

ax2 = ax1.twinx()

ax2.bar(
    daily_merged["date"],
    daily_merged["epa_fires"],
    color="lightgray",
    alpha=0.6,
    width=0.8
)

ax2.set_ylabel("EPA Fires")

ax1.set_title(
    "Daily Detection Rate and Fire Activity"
)

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
```


    
![png](output_73_0.png)
    


## RQ2


```python
from scipy.spatial import cKDTree
```


```python
active_mask = np.nanmean(finn["pm_tons"].values, axis=0) > 0

finn_lat_grid, finn_lon_grid = np.meshgrid(
    lat,
    lon,
    indexing="ij"
)

finn_points = np.column_stack([
    finn_lon_grid[active_mask],
    finn_lat_grid[active_mask]
])

print("Active FINN cells:", len(finn_points))
```

    Active FINN cells: 96531



```python
finn_points.shape
```




    (96531, 2)




```python
tree = cKDTree(finn_points)
```


```python
epa_points = np.column_stack([
    epa_classified["longitude"],
    epa_classified["latitude"]
])

epa_points.shape
```




    (2771, 2)




```python
dist_deg, idx = tree.query(
    epa_points,
    k=1
)
```


```python
dist_km = dist_deg * 111.0
```


```python
epa_classified["nearest_finn_km"] = dist_km
```


```python
print(
    epa_classified["nearest_finn_km"].describe(
        percentiles=[0.25,0.5,0.75,0.9,0.95]
    )
)
print("Mean:",
      epa_classified["nearest_finn_km"].mean())

print("Median:",
      epa_classified["nearest_finn_km"].median())

print("90th percentile:",
      epa_classified["nearest_finn_km"].quantile(0.9))

print("95th percentile:",
      epa_classified["nearest_finn_km"].quantile(0.95))

```

    count    2771.000000
    mean        9.589242
    std         6.981049
    min         0.149990
    25%         4.662067
    50%         7.739678
    75%        12.726923
    90%        19.406571
    95%        24.029365
    max        44.361892
    Name: nearest_finn_km, dtype: float64
    Mean: 9.58924230895704
    Median: 7.739678231439984
    90th percentile: 19.40657051523029
    95th percentile: 24.029364630981664



```python
fig, ax = plt.subplots(figsize=(7,5))

ax.hist(
    epa_classified["nearest_finn_km"],
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("Nearest FINN Detection Distance (km)")
ax.set_ylabel("EPA Fire Count")
ax.set_title("Distance to Nearest FINN Detection")

plt.tight_layout()
plt.show()
```


    
![png](output_84_0.png)
    



```python
dist_sorted = np.sort(
    epa_classified["nearest_finn_km"]
)

cdf = np.arange(
    1,
    len(dist_sorted)+1
)/len(dist_sorted)

fig, ax = plt.subplots(figsize=(7,5))

ax.plot(dist_sorted, cdf)

ax.set_xlabel(
    "Nearest FINN Detection Distance (km)"
)

ax.set_ylabel(
    "Cumulative Probability"
)

ax.set_title(
    "Cumulative Distribution of EPA–FINN Distances"
)

ax.grid(True)

plt.tight_layout()
plt.show()
```


    
![png](output_85_0.png)
    



```python
fig, ax = plt.subplots(figsize=(6,5))

epa_classified.boxplot(
    column="nearest_finn_km",
    by="matched",
    ax=ax
)

ax.set_xlabel("Matched")
ax.set_ylabel("Distance (km)")

plt.suptitle("")
plt.tight_layout()
plt.show()
```


    
![png](output_86_0.png)
    



Matched fires
↓ shorter distances

Missed fires
↑ larger distances



```python
from scipy.stats import mannwhitneyu

matched = epa_classified.loc[
    epa_classified["matched"] == 1,
    "nearest_finn_km"
]

missed = epa_classified.loc[
    epa_classified["matched"] == 0,
    "nearest_finn_km"
]

u, p = mannwhitneyu(
    matched,
    missed,
    alternative="two-sided"
)

print("Mann-Whitney U:", u)
print("p-value:", p)
```

    Mann-Whitney U: 9911.0
    p-value: 0.0



```python
distance_bins = [
    0,1,5,10,25,50,np.inf
]

distance_labels = [
    "<1",
    "1–5",
    "5–10",
    "10–25",
    "25–50",
    ">50"
]

epa_classified["distance_bin"] = pd.cut(
    epa_classified["nearest_finn_km"],
    bins=distance_bins,
    labels=distance_labels
)
```


```python
distance_summary = (
    epa_classified
    .groupby("distance_bin")
    .size()
)

print(distance_summary)
```

    distance_bin
    <1        39
    1–5      785
    5–10     922
    10–25    910
    25–50    115
    >50        0
    dtype: int64


    /glade/derecho/scratch/spatrin/tmp/ipykernel_91762/921519159.py:3: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      .groupby("distance_bin")



```python

```


```python

```


```python

```
