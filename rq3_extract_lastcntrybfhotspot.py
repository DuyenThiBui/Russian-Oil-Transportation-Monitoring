# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:29:10 2025

@author: Duyen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 02:23:26 2025

@author: Duyen
""" 
import os
cwd = os.getcwd()
#os.chdir('D:\\Dropbox\\Duyen\\University\\Master\\Year 2\\Internship\\')
processes = os.cpu_count() - 2
from itertools import islice
import sys
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
import time

import itertools
from datetime import datetime, timedelta
import networkx as nx
from Code import data_processing as pr
from Code import data_preprocessing as pp
from Code import routefinding as rf
from Code import routefinding_v2 as rf2
from Code import rq3_actrytoeu as rf3
from Code.data_processing import find_matched_imo_at_shared_port_noloop_par
from Code import tracking_time_mem as trtm
import psutil, os, time, threading
from memory_profiler import memory_usage
import numpy as np
import pandas as pd
import psutil
import joblib
import multiprocess
import csv
import pickle

from  multiprocess import Pool
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# Create directory to story data
pr_input_path = './processing/pr_inter_input'
pr_output_path = './processing/pr_inter_output'
try:
    os.makedirs(pr_output_path)
    os.makedirs(pr_input_path)
    print(f" path '{pr_input_path}' created successfully")
    print(f" path '{pr_output_path}' created successfully")
except FileExistsError:
    print(
        f"One or more direcotries in '{pr_input_path}' and '{pr_output_path}' aldready exist")

except PermissionError():
    print(
        f"Permission denied: Unable to create '{pr_input_path}' and '{pr_output_path}'")
except Exception as e:
    print(f"An error occured: {e}")
# %% import data

alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
                                  dtype= {'IMO' : 'int64', 'DepPort':'object',
                                          'ArrPort':'object',
                                          'ShipType':'object',
                                          'Country':'object',
                                          'Arr_Country':'object'}, 
                                  parse_dates= ['DepDate', 'ArrDate'],
                                  index_col = 0).rename_axis('Index')
# alltankers_adjusted = pd.read_csv('./RU_oil_tankers_data.csv',
#                                   dtype= {'IMO' : 'int64', 'DepPort':'object',
#                                           'ArrPort':'object',
#                                           'ShipType':'object',
#                                           'Country':'object',
#                                           'Arr_Country':'object'}, 
#                                   parse_dates= ['DepDate', 'ArrDate'],
#                                   index_col = 0).rename_axis('Index')
# correcting data type
for col in ['TravelTime', 'BerthTime']:
    
    alltankers_adjusted[col] = pd.to_timedelta(alltankers_adjusted[col])

with open("./processing/pr_inter_input/crudeoilstat_lookup.pkl", "rb") as f:
    lookup_crude = pickle.load(f)
with open("./processing/pr_inter_input/refinedoilstat_lookup.pkl", "rb") as f:
    lookup_refined = pickle.load(f)


countries = alltankers_adjusted['Country'].unique()
imo_w_oilstat = pd.DataFrame({
    'Country': countries,
    'crude_status': [next(status for status, group in lookup_crude.items() if country in group) for country in countries],
    'refined_status': [next(status for status, group in lookup_refined.items() if country in group) for country in countries]
})

# %% Port selections for different regions

country_of_interest = ['China', 'India', 'Turkey', 'Singapore',
                       'Egypt', 'Brazil']
ru_country = ['Russia']
eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]
NL = ['Netherlands']
port_of_interest = {}
for ctry in country_of_interest:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    port_of_interest[ctry] = list(ctry_ports)


updated_hotspot_countries = ['China', 'India', 'Turkey', 'Kazakhstan'
                       ]
ports_of_hotspots = {}
for ctry in updated_hotspot_countries:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    ports_of_hotspots[ctry] = list(ctry_ports)
# Select RU port

ru_country = ['Russia']
port_of_russia = pr.extract_ports_based_countries(alltankers_adjusted, ru_country)

a = alltankers_adjusted[alltankers_adjusted['Country'] == 'Russia']['DepPort'].unique()
b = alltankers_adjusted[alltankers_adjusted['Arr_Country'] == 'Russia']['ArrPort'].unique()
ab = list(set(a) | set(b))
eu_ports = pr.extract_ports_based_countries(alltankers_adjusted, eu_countries)
# Select the Dutch ports

NL_ports =  pr.extract_ports_based_countries(alltankers_adjusted, NL)
# %% Creating network and find neighbours and connected IMO
# create network
start_time = time.time()

network_edges = []
for n in range(len(alltankers_adjusted)):
    info = tuple([alltankers_adjusted['DepPort'].iloc[n], 
                  alltankers_adjusted['ArrPort'].iloc[n],
                  {'DepDate' : str(alltankers_adjusted['DepDate'].iloc[n]),
                   'ArrDate' : str(alltankers_adjusted['ArrDate'].iloc[n]),
                   'TravelTime' : str(alltankers_adjusted['TravelTime'].iloc[n]),
                   'IMO': alltankers_adjusted['IMO'].iloc[n]}])
    network_edges.append(info)
# create graph
## multi-direct-graph
Graph_whole_dataset = nx.MultiDiGraph()
Graph_whole_dataset.add_edges_from(network_edges)
# betweeness centrality
btwcentr = nx.betweenness_centrality(Graph_whole_dataset)
# extract betweeness centrality of ports for each country
bwtcentr_ports = []
for ctr_of_int in country_of_interest:
    filter_value = {port: btwcentr[port] for port in port_of_interest[ctr_of_int]}
    # Get top 2 keys with highest values
    top_2_keys = sorted(filter_value, key=filter_value.get, reverse=True)[:5]
    bwtcentr_ports.append(top_2_keys)
bwtcentr_ports = [port for sublist in bwtcentr_ports for port in sublist]


### top 5 ports for each hot spots
btw_ctral_hotpots_ports = []
for ctr_of_int in ports_of_hotspots:
    filter_value = {port: btwcentr[port] for port in ports_of_hotspots[ctr_of_int]}
    # Get top 2 keys with highest values
    top_ports = sorted(filter_value, key=filter_value.get, reverse=True)[:5]
    btw_ctral_hotpots_ports.append(top_ports)
btw_ctral_hotpots_ports = [port for sublist in btw_ctral_hotpots_ports for port in sublist]

# extract all the values out
all_hotspots_ports = list(ports_of_hotspots.values())
all_hotspots_ports = [port for sublist in all_hotspots_ports for port in sublist]
# extract ports of NL and RU with the highest betweeness centrality
start_RU_port = list(set(bwtcentr_ports) & set(port_of_russia))
end_port = port_of_russia
# remove RU and NL port from the port lists of highest betweeness centrality
# for port in start_RU_port:
#     bwtcentr_ports.remove(port)
# for port in end_port:
#     bwtcentr_ports.remove(port)

start_RU_port = all_hotspots_ports

#start_RU_port = NL_ports
end_port = port_of_russia
# getting ports of all hotspots

# %% find routes made by 1 IMO
# extract routes from all RU ports to all hotpot ports
# only 2 ports
start_time = time.time()
upperbound_time = float('inf')
lowerbound_time = 0
win_time_slide = 1
strike = 'None'
RU_to_NL = False
loop = True
loop_type = 'country'
IMO_con = True
RU_to_NL_con = False
port_of_interest = bwtcentr_ports
nr_imo = 1
outputpath = f'./processing/pr_inter_output/test9.joblib'
# Round 1

# start iterating

final_route_RU_to_NL = rf3.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                  1, 1, 
                  strike,6, nr_imo, outputpath, Graph_whole_dataset, 
                  port_of_russia, eu_ports, port_of_interest,
                  alltankers_adjusted, ru_country, RU_to_NL, RU_to_NL_con, IMO_con, loop, loop_type)

hottoru = joblib.load('./processing/pr_inter_output/test9.joblib')
min_size = min(len(sublist) for sublist in hottoru)
max_size = max(len(sublist) for sublist in hottoru)
hottoru_v1 = []
for nr_port in range(min_size+1, max_size+2):
    
    routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
        hottoru, alltankers_adjusted, nr_port, 1,  False, oiltype = 'all', loop_type = 'country')
    hottoru_v1.append(routes) 

incl_nl_in_routes = []
for lst in hottoru_v1[1:]:
    for df in lst:
        if (df['Arr_Country'] == 'Netherlands').any():
            incl_nl_in_routes.append(df)


lst_cntry_is_NL_bf_RU = []

for df in incl_nl_in_routes:
    if (df.iloc[-1]['Country'] == 'Netherlands'):
        lst_cntry_is_NL_bf_RU.append(df)
len(incl_nl_in_routes)
len(lst_cntry_is_NL_bf_RU)
################################ last country before RU count from hotspot to RU with nr of total ports >=4 IMPORTANCE
last_country_bf_RU_gen = []
for lst in hottoru_v1[1:]:
    for df in lst:

            last_country_bf_RU_gen.append(df.iloc[[-1]])
last_country_bf_RU_gen = pd.concat(last_country_bf_RU_gen)
a_count = collections.Counter(last_country_bf_RU_gen['Country'])

# assign oil stat

hottoru_v2 = list(chain.from_iterable(hottoru_v1[1:]))
hottoru_v2_w_stat = []
for df in hottoru_v2:
    df = pd.merge(df, imo_w_oilstat, on='Country')
    hottoru_v2_w_stat.append(df)
hottoru_v2_w_empfull = []
for df in hottoru_v2_w_stat:
    df['tankers_status'] = 'nan'
    
    if 'Crude Oil Tanker' in df.iloc[0]['ShipType']:
        if df.iloc[-2]['crude_status'] == 'Net crude importer':
            df.at[df.index[-1], 'tankers_status'] = 'empty'
            df.at[df.index[-1], 'oil_final_status'] = 'Net crude importer'
        else:
            df.at[df.index[-1], 'tankers_status'] = 'full'
            df.at[df.index[-1], 'oil_final_status'] = 'Net crude exporter'
    else:
        if df.iloc[-2]['refined_status'] == 'Net ref. importer':
            df.at[df.index[-1], 'tankers_status'] = 'empty'
            df.at[df.index[-1], 'oil_final_status'] = 'Net ref. importer'
        else:
            df.at[df.index[-1], 'tankers_status'] = 'full'
            df.at[df.index[-1], 'oil_final_status'] = 'Net ref. exporter'
    hottoru_v2_w_empfull.append(df)
# extract last row with state of full or empty
lst_row = []
for df in hottoru_v2_w_empfull:
    lastrow = df.iloc[[-1]]
    lst_row.append(lastrow)
cntry_status_hottoru = pd.concat(lst_row)


grouped = (
    cntry_status_hottoru
    .groupby(['Country', 'tankers_status', 'oil_final_status'])
    .size()
    .reset_index(name='count')
)
# Example: assume df has Country, oil_final_status, tankers_status, count
# Pivot with both oil_final_status and tankers_status


# --- Step 1: compute top 12 countries by total count ---
country_totals = grouped.groupby("Country")["count"].sum()
top12_countries = country_totals.nlargest(12).index

# --- Step 2: split into top12 and "Other" ---
df_top = grouped[grouped["Country"].isin(top12_countries)]
df_other = grouped[~grouped["Country"].isin(top12_countries)]

# Aggregate "Other" into one row
df_other_agg = (
    df_other.groupby(["oil_final_status", "tankers_status"], as_index=False)["count"]
    .sum()
)
df_other_agg["Country"] = "Other"

# Combine
df_combined = pd.concat([df_top, df_other_agg], ignore_index=True)

# --- Step 3: pivot with oil status + tanker status ---
pivoted = df_combined.pivot_table(
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
pivoted = pivoted.drop(columns="total")

# --- Step 4: plot stacked horizontal bar ---
fig, ax = plt.subplots(figsize=(16, 12))

left = pd.Series([0] * len(pivoted), index=pivoted.index)

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

    ax.barh(
        pivoted.index,
        pivoted[col],
        left=left,
        label=col,
        color=color,
        edgecolor="black",
        hatch=hatch
    )
    left += pivoted[col]

# --- Step 5: format ---

ax.set_ylabel('Country', fontsize=40)
ax.set_xlabel('Count', fontsize=40)
plt.yticks(fontsize=40)
plt.xticks(fontsize=30)
plt.legend(title='Oil trade status', fontsize=25, title_fontsize=25)

plt.tight_layout()
plt.show()
plt.savefig('./screenshots/lastcntryfromhottoRU.pdf', format='pdf')
############################### IMPORTANCE
alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
                                  dtype= {'IMO' : 'int64', 'DepPort':'object',
                                          'ArrPort':'object',
                                          'ShipType':'object',
                                          'Country':'object',
                                          'Arr_Country':'object'}, 
                                  parse_dates= ['DepDate', 'ArrDate'],
                                  index_col = 0).rename_axis('Index')
routes_w_2ports_1imo = joblib.load('./processing/pr_inter_output/potential_routes_allRUport__timeinf_nrtotport2.joblib')
routes_w_loop_4w = joblib.load('./processing/pr_inter_output/potential_routes_loop_nrRU_1_time4w_nrtotport4.joblib')
routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    routes_w_2ports_1imo, alltankers_adjusted, 2, 1,  False, oiltype = 'all', loop_type = 'country')
routes4,trip_freq_dict4, country_seq4, port_sequence4 = pr.route_seq_matched_nrimo_par(
    routes_w_loop_4w, alltankers_adjusted, 4, 3,  False, oiltype = 'all', loop_type = 'country')



import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# --- Your data (counters per country) ---
Egp_port_count = Counter({'Crude Oil Tanker': 39,
                          'Products Tanker': 20,
                          'Crude/Oil Products Tanker': 18,
                          'Ore/Oil Carrier': 2,
                          'Chemical Tanker': 1})

sing_port_count = Counter({'Crude/Oil Products Tanker': 26,
                           'Crude Oil Tanker': 20,
                           'Products Tanker': 11})

Baz_port_count = Counter({'Products Tanker': 55,
                          'Crude/Oil Products Tanker': 22})

Kaz_port_count = Counter({'Products Tanker': 265})

Ind_port_count = Counter({'Crude Oil Tanker': 453,
                          'Crude/Oil Products Tanker': 101,
                          'Products Tanker': 25,
                          'Chemical Tanker': 4})

Tur_port_count = Counter({'Products Tanker': 201,
                          'Crude Oil Tanker': 133,
                          'Crude/Oil Products Tanker': 56,
                          'Chemical Tanker': 7,
                          'Ore/Oil Carrier': 3,
                          'Chemical/Products Tanker': 1})

China = Counter({'Crude Oil Tanker': 353,
                  'Crude/Oil Products Tanker': 140,
                  'Products Tanker': 33,
                  'Chemical/Products Tanker': 10,
                  'Chemical Tanker': 1})

# --- Step 1: Group tankers into 3 categories ---
def group_tankers(counter):
    grouped = {"Crude Oil Tanker": 0,
               "Crude/Oil Products Tanker": 0,
               "Refined Oil Tankers": 0}
    
    for k, v in counter.items():
        if k == "Crude Oil Tanker":
            grouped["Crude Oil Tanker"] += v
        elif k == "Crude/Oil Products Tanker":
            grouped["Crude/Oil Products Tanker"] += v
        else:
            grouped["Refined Oil Tankers"] += v
    return grouped

countries = {
    "Egypt": Egp_port_count,
    "Singapore": sing_port_count,
    "Brazil": Baz_port_count,
    "Kazakhstan": Kaz_port_count,
    "India": Ind_port_count,
    "Turkey": Tur_port_count,
    "China": China
}

# --- Step 2: Build DataFrame ---
grouped_data = {country: group_tankers(counter) for country, counter in countries.items()}
df = pd.DataFrame(grouped_data).T  # transpose so countries are rows

# --- Step 3: Plot ---
ax = df.plot(kind="bar", stacked=True, figsize=(14,9))

plt.xlabel("Countries", fontsize=40)
plt.ylabel("Number of Tankers", fontsize=40)
plt.xticks(rotation=90, ha="right", fontsize=30)
plt.yticks(fontsize=30)
plt.legend(title="Tanker Types", fontsize=30, title_fontsize=25)
plt.tight_layout()
plt.show()

plt.savefig('./screenshots/tankertypefromRUtonextcntr.pdf', format='pdf')