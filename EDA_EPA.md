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
from utils import *
from data_loading import load_epa_data, filter_june
```


```python
df = load_epa_data()
df_june = filter_june(df)

print(f"Total fires: {len(df):,}")
print(f"June fires: {len(df_june):,}")

df.head()
```

    Total fires: 79,563
    June fires: 2,771





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
      <th>id</th>
      <th>event_id</th>
      <th>event_name</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>type</th>
      <th>area</th>
      <th>fips</th>
      <th>state</th>
      <th>...</th>
      <th>74873</th>
      <th>75058</th>
      <th>75070</th>
      <th>79107</th>
      <th>85018</th>
      <th>91203</th>
      <th>95476</th>
      <th>sources</th>
      <th>scc_description</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-01-01</td>
      <td>AG16545_0101</td>
      <td>AG16545_0101</td>
      <td>AG16545_0101</td>
      <td>31.278</td>
      <td>-83.460</td>
      <td>Agricultural</td>
      <td>1.0</td>
      <td>13075</td>
      <td>Georgia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-01-01</td>
      <td>AG13323_0101</td>
      <td>AG13323_0101</td>
      <td>AG13323_0101</td>
      <td>31.590</td>
      <td>-83.887</td>
      <td>Agricultural</td>
      <td>4.0</td>
      <td>13321</td>
      <td>Georgia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-01-01</td>
      <td>AG13278_0101</td>
      <td>AG13278_0101</td>
      <td>AG13278_0101</td>
      <td>34.613</td>
      <td>-83.894</td>
      <td>Agricultural</td>
      <td>1.0</td>
      <td>13187</td>
      <td>Georgia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-01-02</td>
      <td>AG16723_0102</td>
      <td>AG16723_0102</td>
      <td>AG16723_0102</td>
      <td>32.295</td>
      <td>-83.435</td>
      <td>Agricultural</td>
      <td>1.0</td>
      <td>13235</td>
      <td>Georgia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-01-02</td>
      <td>AG7192_0102</td>
      <td>AG7192_0102</td>
      <td>AG7192_0102</td>
      <td>32.474</td>
      <td>-81.705</td>
      <td>Agricultural</td>
      <td>2.0</td>
      <td>13031</td>
      <td>Georgia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
counts = df["type"].value_counts()
counts = enforce_fire_order(counts)

colors = get_fire_colors(counts.index)

fig, ax = plt.subplots()

counts.plot(
    kind="bar",
    color=colors,
    ax=ax
)

add_bar_labels(ax, counts.values)

ax.set_title("Number of Fire Events by Type in Georgia (2022)")
ax.set_xlabel("Fire Type")
ax.set_ylabel("Count")
ax.set_xticklabels(counts.index, rotation=0)

plt.tight_layout()
plt.show()
```


    
![png](output_4_0.png)
    



```python
counts_june = df_june["type"].value_counts()
counts_june = enforce_fire_order(counts_june)

colors = get_fire_colors(counts_june.index)

fig, ax = plt.subplots()

counts_june.plot(
    kind="bar",
    color=colors,
    ax=ax
)

add_bar_labels(ax, counts_june.values)

ax.set_title("Number of Fire Events by Type in Georgia (June 2022)")
ax.set_xlabel("Fire Type")
ax.set_ylabel("Count")
ax.set_xticklabels(counts_june.index, rotation=0)

plt.tight_layout()
plt.show()

counts_june / counts_june.sum() * 100
```


    
![png](output_5_0.png)
    





    type
    Prescribed      37.820281
    Agricultural    35.691086
    Wildfire        26.488632
    Name: count, dtype: float64




```python
monthly = df.groupby(["month", "type"]).size().unstack()
monthly = monthly.reindex(columns=FIRE_TYPES)

fig, ax = plt.subplots(figsize=(10,5))

for fire_type in FIRE_TYPES:
    ax.plot(
        monthly.index,
        monthly[fire_type],
        label=fire_type,
        color=FIRE_COLORS[fire_type],
        linewidth=2
    )

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTHS_LABELS, rotation=45)

ax.set_title("Monthly Fire Trends in Georgia (2022)")
ax.set_xlabel("Month")
ax.set_ylabel("Fire Count")

ax.grid(alpha=0.3)
ax.legend(title="Fire Type")

plt.tight_layout()
fname = os.path.join(FIG_DIR, "monthly_fire_trends.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.show()
```


    
![png](output_6_0.png)
    



```python
plt.figure()

for t in FIRE_TYPES:
    vals = df[df["type"] == t]["pm2.5"].dropna().values
    vals = vals[vals > 0]
    subset = log_transform(vals)
    sns.kdeplot(
        subset,
        label=t,
        color=FIRE_COLORS[t],
        linewidth=2
    )

plt.title("PM2.5 Distribution (log10)")
plt.xlabel("log10(PM2.5)")
plt.ylabel("Density")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](output_7_0.png)
    



```python
plt.figure()

for t in FIRE_TYPES:
    vals = df[df["type"] == t]["pm2.5"].dropna().values
    vals = vals[vals > 0]
    x, y = cumulative_curve(vals)

    plt.plot(
        x,
        y,
        label=t,
        color=FIRE_COLORS[t],
        linewidth=2
    )

plt.xlabel("% Fires")
plt.ylabel("% Emissions")

plt.title("Cumulative PM2.5 Contribution by Fire Type")
plt.legend(title="Fire Type")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    



```python
# monthly PM 2.5 emissions by fire type
monthly_pm25 = df.groupby(["month", "type"])["pm2.5"].sum().unstack()
monthly_pm25 = monthly_pm25.fillna(0)
monthly_pm25 = monthly_pm25.reindex(columns=FIRE_TYPES)

fig, ax = plt.subplots(figsize=(10,5))

for t in FIRE_TYPES:
    ax.plot(
        monthly_pm25.index,
        monthly_pm25[t],
        label=t,
        color=FIRE_COLORS[t],
        linewidth=2
    )

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTHS_LABELS, rotation=45)

ax.set_title("Monthly PM2.5 Emissions by Fire Type")
ax.set_xlabel("Month")
ax.set_ylabel("Total PM2.5 Emissions (tons)")

ax.grid(alpha=0.3)
ax.legend(title="Fire Type")

fname = os.path.join(FIG_DIR, "EPA_monthly_pm25_emissions.png")
plt.savefig(
    fname,
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    


1. Prescribed burns dominate total fire counts in Georgia
2. Agricultural fires are also frequent but slightly lower than prescribed
3. Wildfires occur less often but may have higher variability in emissions
4. PM2.5 emissions are highly skewed across all fire types
5. Emissions are highly concentrated, with a small fraction of fires contributing the majority of PM2.5.
6. Seasonal patterns show strong variation in fire activity throughout the year
7. Agricultural and prescribed fires dominate event counts, while wildfires contribute disproportionately to emissions.



```python
df_june["area_km2"] = df_june["area"] * 0.00404686
```


```python
fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    df_june["pm2.5"],
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("PM2.5 Emissions (tons)")
ax.set_ylabel("Frequency")
ax.set_title("EPA PM2.5 Distribution")

plt.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    



```python
df_june["log_pm25"] = np.log10(df_june["pm2.5"] + 0.001)

plt.figure(figsize=(8,5))
plt.hist(
    df_june["log_pm25"],
    bins=30,
    edgecolor="black")

plt.xlabel("log10(PM2.5 tons)")
plt.ylabel("Frequency")
plt.title("Log PM2.5 Distribution")
plt.show()
```


    
![png](output_13_0.png)
    



```python
plt.figure(figsize=(8,5))

plt.hist(
    df_june["area_km2"],
    bins=40,
    edgecolor="black"
)

plt.xlabel("Burned Area (km²)")
plt.ylabel("Frequency")
plt.title("Burned Area Distribution")
plt.show()
```


    
![png](output_14_0.png)
    



```python
df_june["log_area_km2"] = np.log10(df_june["area_km2"] + 1e-4)

plt.figure(figsize=(8,5))

plt.hist(
    df_june["log_area_km2"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("log10(Burned Area km²)")
plt.ylabel("Frequency")
plt.title("Log Burned Area Distribution")
plt.show()
```


    
![png](output_15_0.png)
    



```python

```
