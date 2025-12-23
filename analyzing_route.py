# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 07:55:49 2025

@author: Duyen

This script generates statistical graphs, diagram, and mapss showing Russian oil
distribution patterns when given different starting and ending points. We 
look into the patterns of both direct and detour routes between given locations:

+ Russian ports to all its direct and indirect trading patners.
+ Hotspot ports (India, Turkey, China, and Kazakhstan) to Dutch ports
+ European and Dutch ports to Russian ports.

When a result shows an expected outcome, we dive deeper into detail to
understand the cause. For example: Vessels often stoped in Greece and Italy 
before arriving in hotspots. We will analyze types of vessels, flag 
nationalities, previous/consecutive trips of related vessels.

All input data of this script was generated from "main" scripts.
"""

# import library
import os
import sys
import collections
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from itertools import chain
from Code import data_processing as pr
from Code import plot as plt_cus
import pickle
import pyproj

# %% setting
# directory
cwd = os.getcwd()
os.chdir(cwd)
# 
pyproj.datadir.set_data_dir(r"C:\Users\Duyen\anaconda3\Library\share\proj")
pd.DataFrame.iteritems = pd.DataFrame.items
# turn off warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %% import data
# vessel data
alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
                                  dtype= {'IMO' : 'int64', 'DepPort':'object',
                                          'ArrPort':'object',
                                          'ShipType':'object',
                                          'Country':'object',
                                          'Arr_Country':'object'}, 
                                  parse_dates= ['DepDate', 'ArrDate'],
                                  index_col = 0).rename_axis('Index')

# correct data type
for col in ['TravelTime', 'BerthTime']:
    
    alltankers_adjusted[col] = pd.to_timedelta(alltankers_adjusted[col])
    
# import lookup table for oil trade status
with open("./processing/pr_inter_input/crudeoilstat_lookup.pkl", "rb") as f:
    lookup_crude = pickle.load(f)
with open("./processing/pr_inter_input/refinedoilstat_lookup.pkl", "rb") as f:
    lookup_refined = pickle.load(f)
    
    
# from RU to hotspots
# import data from all RU to all port except EU
routes_w_2ports_1imo = joblib.load('./processing/pr_inter_output/potential_routes_allRUport_to_allportexceptEUport__timeinf_nrtotport2.joblib')
# import data from RU to only EU ports
routes_w_2ports_1imo_to_allEuport = joblib.load('./processing/pr_inter_output/potential_routes_allRUport_to_allEUport__timeinf_nrtotport2.joblib')
# import data from all RU ports to all hotspots
#routes_w_2ports_1imo = joblib.load('./processing/pr_inter_output/potential_routes_allRUport__timeinf_nrtotport2.joblib')



# %% new update for RQ 3
# preprocess
# determine hotspot countries and hotspot ports   
    
hotspot_countries = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Malaysia','Kazakhstan','Egypt','EU countries']
hotspot_ports = ['Sikka', 'Vadinar Terminal', 'Dongying', 'Dongjiakou', 'Yarimca',
                 'Aliaga', 'Singapore', 'Trieste', 'STS Lanconia Bay', 'Rotterdam', 
                   'Aktau']

# route = joblib.load('./processing/pr_inter_output/potential_routes_loop_nrRU_1_time1m_nrtotport5.joblib')

countries = alltankers_adjusted['Country'].unique()

eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]

eu_ports = pr.extract_ports_based_countries(alltankers_adjusted, eu_countries)


# assign oil status for each country
imo_w_oilstat = pd.DataFrame({
    'Country': countries,
    'crude_status': [next(status for status, group in lookup_crude.items() if country in group) for country in countries],
    'refined_status': [next(status for status, group in lookup_refined.items() if country in group) for country in countries]
})

# Extract all countries except EU
all_countries_except_EU = set(alltankers_adjusted['Country'].unique()) - set(eu_countries)
port_all_cntr_except_EU = [
    port
    for ctry in all_countries_except_EU
    for port in alltankers_adjusted.loc[alltankers_adjusted['Country'] == ctry, 'DepPort'].unique()
]
# all routes from RU to hotspots

routes_w_2ports_1imo = list(chain.from_iterable(routes_w_2ports_1imo))

routes_w_2ports_1imo_df = []
for lst in routes_w_2ports_1imo:

    df = alltankers_adjusted.loc[[lst]]
    routes_w_2ports_1imo_df.append(df)
routes_w_2ports_1imo_merged = pd.concat(routes_w_2ports_1imo_df)

# coordinates of main Russian ports. The order follow the order of 'node of poi'
lat = [55.33645166497715, 42.989908134271026,47.246338588377924, 44.72386634131215,
       57.981331706627145, 44.573629154418455, 45.32106723900586, 46.725491744549785, 42.87163669325678,
       42.81172875512808, 45.339389368205545, 45.113947656636405, 60.63202101823246,
       42.71301438443517, 44.09,68.97, 35.595,  22.4604,  40.77, 43.600,   38.087, 38.825]
long = [19.621304440518394,47.48316745254493,38.926908705600475, 37.76724920658686,
        20.46329611924812, 38.02913297585141, 37.38361971433019, 38.27438238234953,
        131.392095298844, 132.88112820333495, 36.674076920849615, 36.7506273905618,
        28.56671176348224, 133.01331187609614,     39.07,  33.04,  119.807,
        69.8180,  29.73, 51.217, 118.964,   26.956 ]


# %% Process
# create geppackage for routes from RU ports to the primary ports of hotspots
RU_direct = collections.Counter(routes_w_2ports_1imo_merged['Country'])
RU_imo_direct = routes_w_2ports_1imo_merged.groupby(["DepPort", "IMO"]).size().unstack(fill_value=0)
RU_imo_direct['Nr of imo contr'] = (RU_imo_direct !=0).sum(axis =1)
RU_imo_direct = RU_imo_direct['Nr of imo contr']

RU_direct_arr = collections.Counter(routes_w_2ports_1imo_merged['ArrPort'])
RU_imo_direct = routes_w_2ports_1imo_merged.groupby(["DepPort", "IMO"]).size().unstack(fill_value=0)
RU_imo_direct['Nr of imo contr'] = (RU_imo_direct !=0).sum(axis =1)
RU_imo_direct = RU_imo_direct['Nr of imo contr']

# extract related routes to each hotspot nation
China = routes_w_2ports_1imo_merged[(routes_w_2ports_1imo_merged['Country'] == 'Russia') &
                                     (routes_w_2ports_1imo_merged['ArrPort'].isin(['Dongjiakou', 'Dongying']))]
China = China[['DepPort', 'ArrPort', 'Country', 'Arr_Country']]
Turkey = routes_w_2ports_1imo_merged[(routes_w_2ports_1imo_merged['Country'] == 'Russia') &
                                     (routes_w_2ports_1imo_merged['ArrPort'].isin(['Yarimca', 'Aliaga']))]
Turkey = Turkey[['DepPort', 'ArrPort', 'Country', 'Arr_Country']]
India = routes_w_2ports_1imo_merged[(routes_w_2ports_1imo_merged['Country'] == 'Russia') &
                                     (routes_w_2ports_1imo_merged['ArrPort'].isin(['Sikka']))]
India = India[['DepPort', 'ArrPort', 'Country', 'Arr_Country']]
Kazakhstan = routes_w_2ports_1imo_merged[(routes_w_2ports_1imo_merged['Country'] == 'Russia') &
                                     (routes_w_2ports_1imo_merged['ArrPort'].isin(['Aktau']))]
Kazakhstan = Kazakhstan[['DepPort', 'ArrPort', 'Country', 'Arr_Country']]
poi_map = pd.concat([China,Turkey, India,Kazakhstan])
depport = list(set(poi_map['DepPort']))
arrport = list(set(poi_map['ArrPort']))
node_of_poi = depport + arrport

node_df = pd.DataFrame({'ID': node_of_poi, 'Label' :node_of_poi, 'Latitude' : lat, 'Longitude' : long})
poi_map = poi_map[['DepPort', 'ArrPort']]
poi_map = poi_map.rename(columns = {'DepPort':'Source', 'ArrPort':'Target'})
# export
node_df.to_csv('./processing/pr_inter_input/node_poi_all.csv', index = True)
poi_map.to_csv('./processing/pr_inter_input/edge_poi_all.csv')

# %% Analyzing patterns from RU directly to its importing partners
# making parallel diagram to illustrate the flows

# from RU to only EU
routes_w_2ports_1imo_to_allEuport = list(chain.from_iterable(routes_w_2ports_1imo_to_allEuport))
routes_w_2ports_1imo_to_allEuport_df = []
for lst in routes_w_2ports_1imo_to_allEuport:

    df = alltankers_adjusted.loc[[lst]]
    routes_w_2ports_1imo_to_allEuport_df.append(df)
routes_w_2ports_1imo_to_allEuport_merged = pd.concat(routes_w_2ports_1imo_to_allEuport_df)

count_Eu_arr_Port = collections.Counter(routes_w_2ports_1imo_to_allEuport_merged['ArrPort'])
# merge data from RU -> only EU and RU -> nonEU countries
route_RU_to_all_hotspot_2p_1imo = pd.concat([routes_w_2ports_1imo_merged,
                                            routes_w_2ports_1imo_to_allEuport_merged])
# group less common ports into a catagory called 'mixed ports'
RU_top5 = (collections.Counter(route_RU_to_all_hotspot_2p_1imo['DepPort'])).most_common(4)
RU_topportnames = [key for key,val in RU_top5]
mask_notmatched_RUnames = ~route_RU_to_all_hotspot_2p_1imo['DepPort'].isin(RU_topportnames)
route_RU_to_all_hotspot_2p_1imo.loc[mask_notmatched_RUnames, 'DepPort'] = 'mixed ports'
# change all Eu countries to Eu
mask_eu_country = route_RU_to_all_hotspot_2p_1imo['Arr_Country'].isin(eu_countries)
route_RU_to_all_hotspot_2p_1imo.loc[mask_eu_country, 'Arr_Country'] = 'EU countries'
mask_other_countries = ~route_RU_to_all_hotspot_2p_1imo['Arr_Country'].isin(hotspot_countries)
route_RU_to_all_hotspot_2p_1imo.loc[mask_other_countries, 'Arr_Country'] = 'Mixed countries'
mask_mixed = route_RU_to_all_hotspot_2p_1imo['Arr_Country'] == 'Mixed countries'
# Update 'Arr_Port' to 'Mixed ports' for those rows
route_RU_to_all_hotspot_2p_1imo.loc[mask_mixed, 'ArrPort'] = 'mixed ports'      
# change arrival ports
top_hotspots_arrport = collections.Counter(RU_direct_arr).most_common(15)
name_sel_tophotspot = [key for key,val in top_hotspots_arrport]
top_Eu_arriveports = collections.Counter(count_Eu_arr_Port).most_common(3)
name_sel_euport = [key for key,val in top_Eu_arriveports]

name_sel_arr_ports = name_sel_tophotspot + name_sel_euport


mask_machted_arrport = ~route_RU_to_all_hotspot_2p_1imo['ArrPort'].isin(hotspot_ports)
route_RU_to_all_hotspot_2p_1imo.loc[mask_machted_arrport, 'ArrPort'] = 'mixed ports'

mix_ports = ['China (mix)', 'India (mix)', 'Turkey (mix)', 'United Arab Emirates (mix)', 'Singapore (mix)',
                       'Saudi Arabia (mix)', 'Korea (South) (mix)', 'Brazil (mix)',
                       'Malaysia (mix)', 'Kazakhstan (mix)', 'Egypt (mix)',  'Eu (mix)', 'Mixed countries (mixed)']
hotspot_countries_update = ['China',
 'India',
 'Turkey',
 'United Arab Emirates',
 'Singapore',
 'Saudi Arabia',
 'Korea (South)',
 'Brazil',
 'Malaysia',
 'Kazakhstan',
 'Egypt',
 'EU countries', 'Mixed countries']
for ind, ctry in enumerate(hotspot_countries_update):
    mask = route_RU_to_all_hotspot_2p_1imo['Arr_Country'] == ctry
    mixed_port_row = route_RU_to_all_hotspot_2p_1imo['ArrPort'] == 'mixed ports'
    route_RU_to_all_hotspot_2p_1imo.loc[mask & mixed_port_row, 'ArrPort'] = mix_ports[ind]

# Visualize
# Force it to render in browser
plt_cus.par_plot(route_RU_to_all_hotspot_2p_1imo,
             "Oil distribution directly from Russian ports to ports of interest",
             'Arr. Countries')
  

# %% Analyzing pattern from RU to hotspot on INDIRECT routes IMPORTANCE
routes_w_maxports_1imo = joblib.load('./processing/pr_inter_output/potential_routes_allRUport_to_allhotport__timeinf_nrtotport5.joblib')
routes_w_3ports_1imo,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    routes_w_maxports_1imo, alltankers_adjusted, 3, 1,  False, oiltype = 'all', loop_type = 'country')
routes_w_4ports_1imo,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    routes_w_maxports_1imo, alltankers_adjusted, 4, 1,  False, oiltype = 'all', loop_type = 'country')
routes_w_5ports_1imo,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    routes_w_maxports_1imo, alltankers_adjusted, 5, 1,  False, oiltype = 'all', loop_type = 'country')
routes_w_6ports_1imo,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    routes_w_maxports_1imo, alltankers_adjusted, 6, 1,  False, oiltype = 'all', loop_type = 'country')
mix_portnr_w_1imo = [routes_w_3ports_1imo, routes_w_4ports_1imo, routes_w_5ports_1imo, routes_w_6ports_1imo]

hotspot_countries_ = ['China', 'Turkey', 'India', 'Kazakhstan' ]

hotspot_countries_crude = ['China', 'Turkey', 'India', 'Kazakhstan' ]
# extract full routes with details
more_than_2port_1iom = []
for lst in mix_portnr_w_1imo:
    route = []
    for df in lst:
        sel_row = df.iloc[1:]
        #if (sel_row['Country'].iloc[0] in hotspot_countries) | (sel_row['Country'].iloc[0] in ['Russia']):
        if (sel_row['Country'].iloc[0] in ['Russia'])|(df['Country'].iloc[1] == df['Arr_Country'].iloc[1])|(sel_row['Country'].iloc[0] in hotspot_countries_crude) :
            next
        else:
            route.append(sel_row)
    more_than_2port_1iom.append(route)
more_than_2port_1iom = [pd.concat(lst) for lst in more_than_2port_1iom]

# assign length of each routes

for ind, _ in enumerate(more_than_2port_1iom):

    more_than_2port_1iom[ind]['nr_port'] = f'ports of {ind+3}'
 
more_than_2port_1iom = pd.concat(more_than_2port_1iom)

# make change long names to abbreviations
more_than_2port_1iom['Country'] = more_than_2port_1iom['Country'].replace({
    "United Kingdom": "UK",
    "United States of America": "USA",
    "United Arab Emirates": "UAE",
    "Chinese Taipei (Taiwan)": 'Taiwan'
})

# plt_cus.stackbar_w_gr_plot(more_than_2port_1iom, 'Country','nr_port',
#                        "Top 10 countries visted by imos before reaching the determined hotspots", 10,
#                        "Counts", "Countries")

# visualize frequency visits of intermediate ports on indirect routes
# and also showing tanker type distributions to each countries

fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)  # 2 rows, 1 column

# --- First plot: group by nr_port ---
df1 = more_than_2port_1iom.groupby(['Country','nr_port']).size().unstack(fill_value=0)
df1['sum'] = df1.sum(axis=1)
df1 = df1.sort_values(by='sum', ascending=False)

top1 = df1[:10]
others_sum = df1[10:].sum(axis=0)
other_df = pd.DataFrame([others_sum], index=['Others'])
top1 = pd.concat([top1, other_df]).drop(columns='sum')

top1.plot(kind="bar", stacked=True, ax=axes[0])
axes[0].set_ylabel('Counts', fontsize=40)
axes[0].tick_params(axis='x', rotation=90, labelsize=30)
axes[0].tick_params(axis='y', labelsize=30)
axes[0].legend(
    fontsize=20,
    loc='upper right',
    bbox_to_anchor=(0.9, 0.9),  # move slightly inward
    frameon=True
)


# --- Second plot: group by Ship Type ---
df2 = more_than_2port_1iom.groupby(['Country','ShipType']).size().unstack(fill_value=0)
# Define which columns go into the merged category
refined_cols = [
    'Chemical Tanker',
    'Chemical/Products Tanker',
    'Ore/Oil Carrier',
    'Products Tanker'
]

# Create the merged 'Refined Oil Products Tanker' column
df2['Refined Oil Tanker'] = df2[refined_cols].sum(axis=1)

# Keep only the required columns
df2 = df2[['Crude Oil Tanker', 'Crude/Oil Products Tanker', 'Refined Oil Tanker']]
df2['sum'] = df2.sum(axis=1)
df2 = df2.sort_values(by='sum', ascending=False)

top2 = df2[:10]
others_sum = df2[10:].sum(axis=0)
other_df = pd.DataFrame([others_sum], index=['Others'])
top2 = pd.concat([top2, other_df]).drop(columns='sum')

top2.plot(kind="bar", stacked=True, ax=axes[1])
axes[1].set_ylabel('Counts', fontsize=40)
axes[1].set_xlabel('Countries', fontsize=45)
axes[1].tick_params(axis='x', rotation=90, labelsize=35)
axes[1].tick_params(axis='y', labelsize=30)
axes[1].legend(
    fontsize=20,
    loc='upper right',
    bbox_to_anchor=(0.9, 0.9),  # move slightly inward
    frameon=True
)


# --- Adjust layout ---
plt.tight_layout()
plt.show()

plt.savefig('./screenshots/shiptypeandroutegroup_indirectRUtohot.pdf', format='pdf')

# Plotting Russian port deliver oil to Italy and GREECE
# require a variable wit list of df

port_Italy = []
port_Greece= []
for lst in mix_portnr_w_1imo:

    for df in lst:
        
        if df['Country'].isin(['Italy']).any() :
            port_Italy.append(df)
        elif df['Country'].isin(['Greece']).any():
            port_Greece.append(df)
RU_port_Italy = []
for df in port_Italy:
    sel = df.iloc[0:1]
    RU_port_Italy.append(sel)
RU_port_Italy = pd.concat(RU_port_Italy)
count_RU_port_Italy = collections.Counter(RU_port_Italy['DepPort'])
RU_port_Greece = []
for df in port_Greece:
    sel = df.iloc[0:1]
    RU_port_Greece.append(sel)
RU_port_Greece = pd.concat(RU_port_Greece)
count_RU_port_Greece = collections.Counter(RU_port_Greece['DepPort'])

# --- Extract top 7 for each ---
top7_Italy = count_RU_port_Italy.most_common(7)
top7_Greece = count_RU_port_Greece.most_common(7)

# --- Convert to lists for plotting ---
ports_Italy, values_Italy = zip(*top7_Italy)
ports_Greece, values_Greece = zip(*top7_Greece)

# --- Determine shared x-axis scale ---
max_value = max(max(values_Italy), max(values_Greece))

# --- Create 1 row × 2 columns figure ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)

# --- Italy subplot ---
axes[0].barh(ports_Italy, values_Italy, color="skyblue", edgecolor="black")
axes[0].set_title("Italy", fontsize=30, fontweight='bold')
axes[0].set_xlabel("Count", fontsize=30)
axes[0].set_ylabel("Port", fontsize=30)
axes[0].tick_params(axis='x', labelsize=30)
axes[0].tick_params(axis='y', labelsize=30)
axes[0].invert_yaxis()  # largest on top
axes[0].set_xlim(0, max_value * 1.1)

# --- Greece subplot ---
axes[1].barh(ports_Greece, values_Greece, color="skyblue", edgecolor="black")
axes[1].set_title(" Greece", fontsize=30, fontweight='bold')
axes[1].set_xlabel("Count", fontsize=30)
axes[1].tick_params(axis='x', labelsize=30)
axes[1].tick_params(axis='y', labelsize=30)
axes[1].invert_yaxis()
axes[1].set_xlim(0, max_value * 1.1)

plt.tight_layout()
plt.show()

plt.savefig('./screenshots/GreandItaPort.pdf', format='pdf')






