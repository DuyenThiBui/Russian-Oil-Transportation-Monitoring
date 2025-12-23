# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:40:24 2025

@author: Duyen
This script generates the results of the Russian oil flow patterns from Hotsport ports to all EU ports, with a partivular focus on the Netherlands.
It provides graphs, and diagrams for direct and indirect routes
It also provides in depth analysis on unexpected outputs obtained from a big picture.
"""

import os
cwd = os.getcwd()
os.chdir(cwd)
import sys
import collections
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from itertools import chain
from Code import data_processing as pr
from Code import plot as plt_cus
import geopandas as gpd
import pickle
from shapely.geometry import Point
import pyproj
print(pyproj.datadir.get_data_dir())
pyproj.datadir.set_data_dir(r"C:\Users\Duyen\anaconda3\Library\share\proj")
pd.DataFrame.iteritems = pd.DataFrame.items
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %% import data

alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
                                  dtype= {'IMO' : 'int64', 'DepPort':'object',
                                          'ArrPort':'object',
                                          'ShipType':'object',
                                          'Country':'object',
                                          'Arr_Country':'object'}, 
                                  parse_dates= ['DepDate', 'ArrDate'],
                                  index_col = 0).rename_axis('Index')

# correcting data type
for col in ['TravelTime', 'BerthTime']:
    
    alltankers_adjusted[col] = pd.to_timedelta(alltankers_adjusted[col])
hotspot_countries = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Malaysia','Kazakhstan','Egypt','EU countries']
hotspot_ports = ['Sikka', 'Vadinar Terminal', 'Dongying', 'Dongjiakou', 'Yarimca',
                 'Aliaga', 'Singapore', 'Trieste', 'STS Lanconia Bay', 'Rotterdam', 
                   'Aktau']

route = joblib.load('./processing/pr_inter_output/potential_routes_loop_nrRU_1_time1m_nrtotport5.joblib')

countries = alltankers_adjusted['Country'].unique()
# import lookup table for oil trade status
with open("./processing/pr_inter_input/crudeoilstat_lookup.pkl", "rb") as f:
    lookup_crude = pickle.load(f)
with open("./processing/pr_inter_input/refinedoilstat_lookup.pkl", "rb") as f:
    lookup_refined = pickle.load(f)


imo_w_oilstat = pd.DataFrame({
    'Country': countries,
    'crude_status': [next(status for status, group in lookup_crude.items() if country in group) for country in countries],
    'refined_status': [next(status for status, group in lookup_refined.items() if country in group) for country in countries]
})


eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]

eu_ports = pr.extract_ports_based_countries(alltankers_adjusted, eu_countries)
all_countries_except_EU = set(alltankers_adjusted['Country'].unique()) - set(eu_countries)
port_all_cntr_except_EU = [
    port
    for ctry in all_countries_except_EU
    for port in alltankers_adjusted.loc[alltankers_adjusted['Country'] == ctry, 'DepPort'].unique()
]
hotspot_countries_ = ['China', 'Turkey', 'India', 'Kazakhstan' ]

# %% Route from hotspots to NL IMPORTANCE
# direct 
hotspot_to_NL_w_2ports_1imo = joblib.load('./processing/pr_inter_output/potential_routes_allhotspot_to_allNLport__timeinf_nrtotport2.joblib')

# indrect routes
hotspot_to_NL_w_10ports_1imo = joblib.load(f'./processing/pr_inter_output/potential_routes_allhotspot_to_allNLport__timeinf_nrtotport10_v1.joblib')

hotspot_to_NL_w_2ports_1imo = list(chain.from_iterable(hotspot_to_NL_w_2ports_1imo))
# extract full indirect routes
min_size = min(len(sublist) for sublist in hotspot_to_NL_w_10ports_1imo)
max_size = max(len(sublist) for sublist in hotspot_to_NL_w_10ports_1imo)
route_hotspot_to_NL_v1 = []
for nr_port in range(min_size+1, max_size+2):
    
    routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
        hotspot_to_NL_w_10ports_1imo, alltankers_adjusted, nr_port, 1,  False, oiltype = 'all', loop_type = 'country')
    route_hotspot_to_NL_v1.append(routes) 
    
# extract full direct routes

hotspot_to_NL_w_2ports_1imo_df = []
for lst in hotspot_to_NL_w_2ports_1imo:

    df = alltankers_adjusted.loc[[lst]]
    hotspot_to_NL_w_2ports_1imo_df.append(df)
hotspot_to_NL_w_2ports_1imo_merged = pd.concat(hotspot_to_NL_w_2ports_1imo_df)

more_than_2port_1iom_hotspot_to_NL = []
for lst in route_hotspot_to_NL_v1:
    route = []
    for df in lst:
        sel_row = df.iloc[1:]
        route.append(sel_row)
    more_than_2port_1iom_hotspot_to_NL.append(route)
more_than_2port_1iom_hotspot_to_NL = [pd.concat(lst) for lst in more_than_2port_1iom_hotspot_to_NL if len(lst)>0]

# assign length of each routes
for ind, _ in enumerate(more_than_2port_1iom_hotspot_to_NL):

    more_than_2port_1iom_hotspot_to_NL[ind]['nr_port'] = f'ports of {ind+3}'

more_than_2port_1iom_hotspot_to_NL = pd.concat(more_than_2port_1iom_hotspot_to_NL)
# plot all countries in between 
plt_cus.stackbar_w_gr_plot(more_than_2port_1iom_hotspot_to_NL, 'Country', 'nr_port',
                       "Top 10 countries visted by imos before reaching the determined hotspots", 15,
                       "Counts", "Countries", others=False)
# %% plot countries right before NL  IMPORTANCE
# NOTE use df with list not concat
ctry_bf_NL_from_hot = []
for lst in more_than_2port_1iom_hotspot_to_NL:
    for df in lst:
        lst_row = df.iloc[[-1]]
        ctry_bf_NL_from_hot.append(lst_row)
ctry_bf_NL_from_hot = pd.concat(ctry_bf_NL_from_hot)
ctry_bf_NL_from_hot_wstat = pd.merge(ctry_bf_NL_from_hot, imo_w_oilstat, on='Country')
ctry_bf_NL_from_hot_wstat.index = ctry_bf_NL_from_hot.index

# Assign tankers status based on oil import/export
ctry_bf_NL_from_hot_wstat['tankers_status'] = 'no status'
ctry_bf_NL_from_hot_wstat['oil_final_status'] = 'no status'
for row in range(len(ctry_bf_NL_from_hot_wstat)):
    poc = ctry_bf_NL_from_hot_wstat.iloc[row]   # now it's a Series

    if 'Crude Oil Tanker' in poc['ShipType']:
        if poc['crude_status'] == 'Net crude importer':
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'tankers_status'] = 'empty'
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'oil_final_status'] = 'Net crude importer'
        else:
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'tankers_status'] = 'full'
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'oil_final_status'] = 'Net crude exporter'
    else:
        if poc['refined_status'] == 'Net ref. importer':
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'tankers_status'] = 'empty'
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'oil_final_status'] = 'Net ref. importer'
        else:
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'tankers_status'] = 'full'
            ctry_bf_NL_from_hot_wstat.at[ctry_bf_NL_from_hot_wstat.index[row], 'oil_final_status'] = 'Net ref. exporter'

grouped = (
    ctry_bf_NL_from_hot_wstat
    .groupby(['Country', 'tankers_status', 'oil_final_status'])
    .size()
    .reset_index(name='count')
)

grouped['Country'] = grouped['Country'].replace({
    "United Kingdom": "UK",
    "United States of America": "USA",
    "United Arab Emirates": "UAE"
})
# Example: assume df has Country, oil_final_status, tankers_status, count
# Pivot with both oil_final_status and tankers_status



# --- Step 1: compute top 10 countries by total count ---
top10_countries = (
    grouped.groupby("Country")["count"]
    .sum()
    .nlargest(20)

    .index
)

top10_countries = (
    top10_countries.to_series()
    .replace({
        "United Kingdom": "UK",
        "United States of America": "USA",
        "United Arab Emirates": "UAE"
    })
    .values   # <-- use values, not index
)

# --- Step 2: filter data to top 10 countries only ---
df_top10 = grouped[grouped["Country"].isin(top10_countries)]

# --- Step 3: pivot with oil status + tanker status ---
pivoted = df_top10.pivot_table(
    index="Country",
    columns=["oil_final_status", "tankers_status"],
    values="count",
    fill_value=0
)

# Flatten MultiIndex columns
pivoted.columns = [f"{status} ({tankers})" for status, tankers in pivoted.columns]
# --- Step 3.5: sort pivoted table by total count descending ---
pivoted["total"] = pivoted.sum(axis=1)
pivoted = pivoted.sort_values("total", ascending=False)
pivoted = pivoted.drop(columns="total")  # remove helper column after sorting

# --- Step 4: plot stacked bar with hatching ---
fig, ax = plt.subplots(figsize=(16, 12))

bottom = pd.Series([0] * len(pivoted), index=pivoted.index)

colors = plt.cm.tab20.colors
color_map = {}

for col in pivoted.columns:
    status, tankers = col.split(" (")
    tankers = tankers.strip(")")

    # consistent color for each oil status
    if status not in color_map:
        color_map[status] = colors[len(color_map) % len(colors)]
    color = color_map[status]

    # hatch if "full"
    hatch = "//" if tankers == "full" else None

    ax.bar(
        pivoted.index,
        pivoted[col],
        bottom=bottom,
        label=col,
        color=color,
        edgecolor="black",
        hatch=hatch
    )
    bottom += pivoted[col]

# --- Step 5: format ---
from matplotlib.ticker import MaxNLocator
ax.set_xlabel('Country', fontsize=40)
ax.set_ylabel('Count', fontsize=40)
plt.xticks(rotation=60, ha='right', fontsize=30)
plt.yticks(fontsize=40)

# Force y-axis ticks to integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Legend inside
plt.legend(
    title='Oil Status',
    fontsize=30,
    title_fontsize=30,
    loc='upper right',
    frameon=True
)


plt.tight_layout()
plt.show()
plt.savefig('./screenshots/lastcntryfromhottoNL.pdf', format='pdf')

# %% plot countries right after NL of tankers depaturing from hotspots IMPORTANT???
# for direct routes
ctry_af_NL = []
for df in hotspot_to_NL_w_2ports_1imo_df:
    imo =  df.iloc[0,0]
    poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    for row in range(len(poc)):
        if poc.iloc[row:row+1].index.values[0] == df.index.values[0]:
            if row <len(poc)-1:
                ctry_af_NL.append(poc.iloc[row+1:row+2])
ctry_af_NL = pd.concat(ctry_af_NL)
count_ctry_af_NL = collections.Counter(ctry_af_NL['Arr_Country'])    
# for indirect routes
hot_to_NL_indirect = list(chain.from_iterable(route_hotspot_to_NL_v1))
ctry_af_NL_indirect = []
for seg in hot_to_NL_indirect:
    df = seg.iloc[-1:]
    imo =  df.iloc[0,0]
    poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    for row in range(len(poc)):
        if poc.iloc[row:row+1].index.values[0] == df.index.values[0]:
            if row <len(poc)-1:
                ctry_af_NL_indirect.append(poc.iloc[row+1:row+2])
ctry_af_NL_indirect = pd.concat(ctry_af_NL_indirect)
count_ctry_af_NL_indirect= collections.Counter(ctry_af_NL_indirect['Arr_Country'])
# %% routes from all hotspot to EU
# direct
route_allhot_toEu_dir = joblib.load('./processing/pr_inter_output/potential_routes_allhotspot_to_allEUport__timeinf_nrtotport2.joblib')
# indirect
route_allhot_toEu_indir = joblib.load('./processing/pr_inter_output/potential_routes_allhotspot_to_allEUport__timeinf_nrtotport7.joblib')

route_allhot_toEu_dir = list(chain.from_iterable(route_allhot_toEu_dir))


min_size = min(len(sublist) for sublist in route_allhot_toEu_indir)
max_size = max(len(sublist) for sublist in route_allhot_toEu_indir)
route_allhot_toEu_indir_v1 = []
for nr_port in range(min_size+1, max_size+2):
    
    routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
        route_allhot_toEu_indir, alltankers_adjusted, nr_port, 1,  False, oiltype = 'all', loop_type = 'country')
    route_allhot_toEu_indir_v1.append(routes) 

route_allhot_toEu_dir_df = []
for lst in route_allhot_toEu_dir:

    df = alltankers_adjusted.loc[[lst]]
    route_allhot_toEu_dir_df.append(df)

ctry_af_EU = []
for df in route_allhot_toEu_dir_df:
    imo =  df.iloc[0,0]
    poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    for row in range(len(poc)):
        if poc.iloc[row:row+1].index.values[0] == df.index.values[0]:
            if row <len(poc)-1:
                ctry_af_EU.append(poc.iloc[row+1:row+2])
ctry_af_EU = pd.concat(ctry_af_EU)
ctry_af_EU = collections.Counter(ctry_af_EU['Arr_Country'])    
# for indirect routes
hot_to_EU_indirect = list(chain.from_iterable(route_allhot_toEu_indir_v1))
ctry_af_EU_indirect = []
for seg in hot_to_EU_indirect:
    df = seg.iloc[-1:]
    imo =  df.iloc[0,0]
    poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    for row in range(len(poc)):
        if poc.iloc[row:row+1].index.values[0] == df.index.values[0]:
            if row <len(poc)-1:
                ctry_af_EU_indirect.append(poc.iloc[row+1:row+2])
ctry_af_EU_indirect = pd.concat(ctry_af_EU_indirect)
count_ctry_af_EU_indirect= collections.Counter(ctry_af_EU_indirect['Arr_Country'])

# --- Apply renaming ---
count_ctry_af_NL = pr.rename_countries(count_ctry_af_NL)
count_ctry_af_NL_indirect = pr.rename_countries(count_ctry_af_NL_indirect)
ctry_af_EU = pr.rename_countries(ctry_af_EU)
count_ctry_af_EU_indirect = pr.rename_countries(count_ctry_af_EU_indirect)

# --- Prepare counters and titles ---
counters = [
    ("A) After NL (Direct)", count_ctry_af_NL),
    ("B) After NL (Indirect)", count_ctry_af_NL_indirect),
    ("C) After EU (Direct)", ctry_af_EU),
    ("D) After EU (Indirect)", count_ctry_af_EU_indirect)
]

# --- Get max value for consistent scaling ---
max_value = max(max(c.values()) for _, c in counters)

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (ax, (title, counter)) in enumerate(zip(axes, counters)):
    top5 = counter.most_common(5)
    countries = [c for c, _ in top5]
    values = [v for _, v in top5]
    
    ax.barh(countries, values, color='skyblue')
    
    # --- Label control ---
    row, col = divmod(i, 2)
    if row == 1:  # only second row → show x-label
        ax.set_xlabel("Count", fontsize=30)
    if col == 0:  # only first column → show y-label
        ax.set_ylabel("Country", fontsize=30)
    
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_xlim(0, max_value * 1.1)
    ax.invert_yaxis()

plt.tight_layout()
plt.show()
plt.savefig('./screenshots/ctryafNLenEUfromHotspot.pdf', format='pdf')

# %% making parallel diagram for hot->NL directly
ports_hotspot = [

    "Sikka",
    "Aliaga",
    "Ceyhan",
    "Qinzhou",

    'Aktau'

]
hotspot_countries = ['China',
 'India',
 'Turkey',
 
 'Kazakhstan',

 'EU countries']
countries_to_remove = ["Brazil", "Egypt", "Singapore"]

hotspot_to_NL_w_2ports_1imo_merged = hotspot_to_NL_w_2ports_1imo_merged[
    ~hotspot_to_NL_w_2ports_1imo_merged["Country"].isin(countries_to_remove)
]
for ind, ctry in enumerate(hotspot_countries):
    mask = hotspot_to_NL_w_2ports_1imo_merged['Country'] == ctry
    mask2 = ~hotspot_to_NL_w_2ports_1imo_merged['DepPort'].isin(ports_hotspot)
    hotspot_to_NL_w_2ports_1imo_merged.loc[mask & mask2, 'DepPort'] = f'{ctry} (mixed)'



plt_cus.par_plot(hotspot_to_NL_w_2ports_1imo_merged,
             "", 
             'Dep. Countries', dim =  True)
# %% making maps showing intermediate ports between hotspots and NL for each hotspot
extract_hotspot_map = [pd.concat(lst) for lst in route_hotspot_to_NL_v1 if len(lst)>0]
extract_hotspot_map = pd.concat(extract_hotspot_map)
dep_country = extract_hotspot_map['Country'].unique() 

china_map = []
for lst in route_hotspot_to_NL_v1:
    for df in lst:
        if df.iloc[0]['Country'] == 'China':
            china_map.append(df.iloc[1:])
china_map = pd.concat(china_map)

india_map = []
for lst in route_hotspot_to_NL_v1:
    for df in lst:
        if df.iloc[0]['Country'] == 'India':
            india_map.append(df.iloc[1:])
india_map = pd.concat(india_map)
turkey_map = []
for lst in route_hotspot_to_NL_v1:
    for df in lst:
        if df.iloc[0]['Country'] == 'Turkey':
            turkey_map.append(df.iloc[1:])
turkey_map = pd.concat(turkey_map)

hotspotsport_to_ports = list(
    set(china_map['DepPort']) | set(india_map['DepPort']) | set(turkey_map['DepPort']))

# Load
with open("./processing/pr_inter_input/ports_coor_lookup.pkl", "rb") as f:
    loaded_ports = pickle.load(f)

country = ['china','india', 'turkey']
hotspot_var = [china_map, india_map, turkey_map]
for nr in range(len(country)):
    cntrymap = hotspot_var[nr]
    cntrymap['coor'] = cntrymap['DepPort'].map(loaded_ports)
    cntrymap['geometry'] = cntrymap['coor'].apply(lambda x: Point(x[1], x[0]))
    cntrymap_gdf = gpd.GeoDataFrame(cntrymap, geometry='geometry')
    cntrymap_gdf.set_crs(epsg=4326, inplace=True)
    cntrymap_gdf.drop(columns=['TravelTime', 'BerthTime', 'coor'], inplace=True)
    output = f'./ViszOutput/{country[nr]}_to_ports.gpkg'
    cntrymap_gdf.to_file(output, layer='ports', driver="GPKG")
    
# all data
all_hotpot = pd.concat([china_map, india_map, turkey_map])
all_hotpot['coor'] = all_hotpot['DepPort'].map(loaded_ports)
all_hotpot['geometry'] = all_hotpot['coor'].apply(lambda x: Point(x[1], x[0]))
all_hotpot_gdf = gpd.GeoDataFrame(all_hotpot, geometry='geometry')
all_hotpot_gdf.set_crs(epsg=4326, inplace=True)
all_hotpot_gdf.drop(columns=['TravelTime', 'BerthTime', 'coor'], inplace=True)
output = './ViszOutput/{allhotspot_to_ports.gpkg'
all_hotpot_gdf.to_file(output, layer='ports', driver="GPKG")

# %% Making paralell diagram from hotspot to EU countries
eu_countries_noNL = [country for country in eu_countries if country != "Netherlands"]
dir_hotspot_EunonNL = alltankers_adjusted[alltankers_adjusted['Country'].isin(hotspot_countries_) & alltankers_adjusted['Arr_Country'].isin(eu_countries_noNL)]
count_hotspot_EUnonNL = collections.Counter(dir_hotspot_EunonNL['Arr_Country'])

top10_keys_hot_eunonNL  = [k for k, v in sorted(count_hotspot_EUnonNL.items(), key=lambda item: item[1], reverse=True)[:6]]
mask_nameforother = ~dir_hotspot_EunonNL['Arr_Country'].isin(top10_keys_hot_eunonNL)
dir_hotspot_EunonNL.loc[mask_nameforother, 'Arr_Country'] = 'Mixed countries'
plt_cus.sim_par_plot(dir_hotspot_EunonNL, 'from hotspot to EU not NL', 'Dep. Countries' )
a_count = collections.Counter(dir_hotspot_EunonNL['ShipType'])

# making paralell diagrame from ... to NL
dir_acountry_NL = alltankers_adjusted[(~alltankers_adjusted['Country'].isin(['Netherlands'])) & (alltankers_adjusted['Arr_Country'].isin(['Netherlands']))]
count_acountr_NL = collections.Counter(dir_acountry_NL['Country'])

top10_keys_actry_NL  = [k for k, v in sorted(count_acountr_NL.items(), key=lambda item: item[1], reverse=True)[:8]]
mask_nameforother = ~dir_acountry_NL['Country'].isin(top10_keys_actry_NL)
dir_acountry_NL.loc[mask_nameforother, 'Country'] = 'Mixed countries'
plt_cus.sim_par_plot(dir_acountry_NL, 'from a country to  NL', 'Dep. Countries' )

# %% plotting bar chart showing the ship types used to deliver oil in different route segments
#NOTE: missing variables can be found in the script 'RUtoHotspots'


# more_than_2port_1iom_hotspot_to_NL need a list 
lst_row_hotspot_nl = []
for lst in more_than_2port_1iom_hotspot_to_NL:
    col = []
    for df in lst:
        lat_row = df.iloc[1:]
        col.append(lat_row)
    lst_row_hotspot_nl.append(col)
lst_row_hotspot_nl = [pd.concat(lst) for lst in lst_row_hotspot_nl]
    
for ind, _ in enumerate(lst_row_hotspot_nl):

    lst_row_hotspot_nl[ind]['nr_port'] = f'ports of {ind+3}'

lst_row_hotspot_nl = pd.concat(lst_row_hotspot_nl)
lst_row_hotspot_nl['tankers_status'] = 'Nan'
for ind in range(len(lst_row_hotspot_nl)):

    df = lst_row_hotspot_nl.iloc[ind]
    if df['Status'] == 'Net importer':
        lst_row_hotspot_nl.loc[ind, 'tankers_status'] = 'empty'
    else:
        lst_row_hotspot_nl.loc[ind, 'tankers_status'] = 'full'
    
lst_row_hotspot_nl = pd.concat(lst_row_hotspot_nl)
lst_row_hotspot_nl = pd.merge(lst_row_hotspot_nl, imo_w_oilstat, left_on='Country', right_on='Country')
gr_lst_row_hotspot_nl=  lst_row_hotspot_nl.groupby(['Country',"nr_port"]).size().unstack(fill_value=0)
# Take the top 10 by total count across rows
top10 = gr_lst_row_hotspot_nl.sum(axis=1).nlargest(10)   # get top 10 row indices
gr_lst_row_hotspot_nl = gr_lst_row_hotspot_nl.loc[top10.index]      # subset the dataframe

plt_cus.stackbar_plot(gr_lst_row_hotspot_nl,
                  "Top 10 countries visted by imos before reaching the determined hotspots",
                  "Counts", "Countries")

################# IMPORTANCE
## count ship type from hot to NL directly and indiect
# Count ship type from RU ->hótpot
# count ship type from hotspot to next port
# count a port to NL
# note: these have to be df
four_hot_ctry = ['China', 'India', 'Turkey', 'Kazakhstan']
lst_row_hotspot_nl = []
for lst in more_than_2port_1iom_hotspot_to_NL:
    col = []
    for df in lst:
        lat_row = df.iloc[[-1]]
        col.append(lat_row)
    lst_row_hotspot_nl.append(col)
lst_row_hotspot_nl = [pd.concat(lst) for lst in lst_row_hotspot_nl]
lst_row_hotspot_nl = pd.concat(lst_row_hotspot_nl)
hotspot_to_NL_w_2ports_1imo_merged = hotspot_to_NL_w_2ports_1imo_merged[hotspot_to_NL_w_2ports_1imo_merged['Country'].isin(four_hot_ctry)]
hot_NL_inendir = pd.concat([hotspot_to_NL_w_2ports_1imo_merged, lst_row_hotspot_nl ])

shiptype_hot_NL_inendir = collections.Counter(hot_NL_inendir['ShipType'])



#################### count ship type from hot to NL directly and indiect IMPORTANCE

# Count ship type from RU ->hótpot
## this one need to be a list of df routes_w_2ports_1imo_df
ru_to_hotindir = []
for lst in more_than_2port_1iom:
    for df in lst:
        ru_to_hotindir.append(df.iloc[[-1]])
ru_to_hotindir = pd.concat(ru_to_hotindir)
routes_w_2ports_1imo_df = pd.concat(routes_w_2ports_1imo_df)
ru_to_hotindirendir_merge = pd.concat([ru_to_hotindir,routes_w_2ports_1imo_df])
ru_to_hotindirendir_merge = ru_to_hotindirendir_merge[ru_to_hotindirendir_merge['Arr_Country'].isin(four_hot_ctry)]
count_ru_to_hotindirendir_merge = collections.Counter(ru_to_hotindirendir_merge['ShipType'])
# count ship type from hotspot to next EU port
eu_countries_noNL = [country for country in eu_countries ]
dir_hotspot_EunonNL = alltankers_adjusted[alltankers_adjusted['Country'].isin(hotspot_countries_) & alltankers_adjusted['Arr_Country'].isin(eu_countries_noNL)]
count_dir_hotspot_EunonNL = collections.Counter(dir_hotspot_EunonNL['ShipType'])
# count a cntry to NL

dir_acountry_NL = alltankers_adjusted[(~alltankers_adjusted['Country'].isin(['Netherlands'])) & (alltankers_adjusted['Arr_Country'].isin(['Netherlands']))]
count_dir_acountry_NL = collections.Counter(dir_acountry_NL['ShipType'])

# count RU to EU
count_RU_EU = collections.Counter(routes_w_2ports_1imo_to_allEuport_merged['ShipType'])


# Convert counters to pandas Series
s1 = pd.Series(shiptype_hot_NL_inendir, name="Hotspots -> NL")
s2 = pd.Series(count_ru_to_hotindirendir_merge, name="RU -> hotspots")
s3 = pd.Series(count_dir_hotspot_EunonNL, name="Hotspots -> EU")
s4 = pd.Series(count_dir_acountry_NL, name="Countries -> NL")
s5 = pd.Series(count_RU_EU, name = 'RU -> EU')
# Combine into a single DataFrame
df = pd.concat([s1, s3, s5, s4, s2], axis=1).fillna(0).astype(int)
df = df.rename(index={
    'Crude/Oil Products Tanker': 'Mixed Tanker',
    'Chemical/Products Tanker': 'Chem/Prod. Tankers'
})
# Plot grouped bar chart with more gap
ax = df.plot(kind="bar", figsize=(13, 10), width=0.7)  # reduce width (default is 0.8–0.9)

# Formatting

ax.set_xlabel("Ship Type", fontsize=40)
ax.set_ylabel("Count", fontsize=40)
plt.xticks(rotation=90, ha="right", fontsize=30)
plt.yticks(fontsize=25)

# Legend outside
plt.legend(
    title="Routes Types",
    fontsize=20,
    title_fontsize=30,
    bbox_to_anchor=(0.5, 0.9),  # move legend to the right
    loc="upper left",
    borderaxespad=0.
)

plt.tight_layout()
plt.show()

plt.savefig("./screenshots/allshiptypes.pdf", format='pdf')
#############################################