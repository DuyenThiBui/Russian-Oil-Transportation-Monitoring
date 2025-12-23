# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:42:12 2025

@author: Duyen
This script generates the results of the Russian oil flow patterns from EU ports to all RU ports, with a partivular focus on the Netherlands.
It provides graphs, and diagrams for direct and indirect routes
It also provides in depth analysis on unexpected outputs obtained from a big picture.
"""
import os
cwd = os.getcwd()
os.chdir(cwd)
import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from itertools import chain
from Code import data_processing as pr
from Code import plot as plt_cus
import pickle
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
# %% From NL to RU direct
# assign oil status on each poc
alltankers_adjusted_w_status = pd.merge(alltankers_adjusted, imo_w_oilstat, on='Country')
alltankers_adjusted_w_status.index = alltankers_adjusted.index

route_NL_to_RU_direct = joblib.load('./processing/pr_inter_output/potential_routes_allNLport_to_allRUport__timeinf_nrtotport2.joblib')
route_NL_to_RU_direct = list(chain.from_iterable(route_NL_to_RU_direct))
# extract trip from NL to RU directly from the original data
RU_NL_tankers = []

for imo in alltankers_adjusted_w_status['IMO'].unique():
    poc = alltankers_adjusted_w_status[alltankers_adjusted_w_status['IMO'] == imo]
    if ((poc['Country'] == 'Netherlands') & (poc['Arr_Country'] == 'Russia')).any():
        RU_NL_tankers.append(poc)
# Assign tankers status based on oil import/export
RU_NL_tankers_status = []
cntry_bf_NL_RU = []
oil_stat = []
oil_stat_NL = []
for lst in RU_NL_tankers:
    lst = lst.copy()  # avoid modifying original DataFrame by reference
    lst['tankers_status'] = 'no status'
    for row in range(len(lst)):
        if (lst.iloc[row]['Country'] == 'Netherlands') and (lst.iloc[row]['Arr_Country'] == 'Russia'):
            cntry_bf_NL_RU.append(lst.iloc[row-1]['Country'])
            
            oil_stat_NL.append(lst.iloc[row-1]['ShipType'])
            if 'Crude' in lst.iloc[row-1]['ShipType']:
                oil_stat.append(lst.iloc[row-1]['crude_status'])
                if lst.iloc[row - 1]['crude_status'] == 'Net importer':
                    lst.at[lst.index[row], 'tankers_status'] = 'empty'
                else:
                    lst.at[lst.index[row], 'tankers_status'] = 'full'
            else:
                oil_stat.append(lst.iloc[row-1]['refined_status'])
                if lst.iloc[row - 1]['refined_status'] == 'Net importer':
                    lst.at[lst.index[row], 'tankers_status'] = 'empty'
                else:
                    lst.at[lst.index[row], 'tankers_status'] = 'full'

    RU_NL_tankers_status.append(lst)

RU_NL_tankers_status_merged = pd.concat(RU_NL_tankers_status)
route_NL_to_RU_direct_df = []
for lst in route_NL_to_RU_direct:

    df = RU_NL_tankers_status_merged.loc[[lst]]
    route_NL_to_RU_direct_df.append(df)
route_NL_to_RU_direct_merged = pd.concat(route_NL_to_RU_direct_df)



    
# visualize top 10 countries with other categories
    
count_cntry_bf_NL_RU = collections.Counter(cntry_bf_NL_RU)
count_cntry_bf_NL_RU = sorted(count_cntry_bf_NL_RU.items(), key=lambda x: x[1], reverse=True)
top_cntry_bf_NL_RU = count_cntry_bf_NL_RU[:10]

others_sum = sum(v for _, v in count_cntry_bf_NL_RU[10:])
top_cntry_bf_NL_RU.append(("Others", others_sum))



data = {'Country':cntry_bf_NL_RU, 'oil stat':oil_stat}
cntry_bf_NL_RU_df = pd.DataFrame(data)
gr_cntry_bf_NL_RU_df = cntry_bf_NL_RU_df.groupby(['Country', 'oil stat']).size().unstack(fill_value=0)

# --- Extract only the top countries ---
top_countries = [k for k, _ in top_cntry_bf_NL_RU]

# Separate countries and counts
countries = [c for c, _ in top_cntry_bf_NL_RU]
counts = [v for _, v in top_cntry_bf_NL_RU]
data = {'Country':countries, 'Count':counts}
df = pd.DataFrame(data = data)
df = pd.merge(df, imo_w_oilstat, on = 'Country')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(countries, counts, color='steelblue')

# Formatting
ax.set_ylabel('Counts', fontsize=30)
ax.set_title('Top countries a IMO visited before arriving to a Dutch port and then going to direct to a Russian port', fontsize=25, pad=30)
ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

plt.tight_layout()
plt.show()


# visualize top 10 and contribution of net import, net export for each country
# Filter grouped DataFrame for top countries
gr_cntry_bf_NL_RU_df = gr_cntry_bf_NL_RU_df.loc[gr_cntry_bf_NL_RU_df.index.isin(top_countries)]

# plot
plt.stackbar_plot(gr_cntry_bf_NL_RU_df,
                  'Top countries a IMO visited before arriving to a Dutch port and then going to direct to a Russian port',
                  "Counts", "Countries")


# Parallel visualization
plt_cus.par_plot(route_NL_to_RU_direct_merged,
         "Oil distribution directly from NL ports to RU ports",
         'Dep. Ports')




###########
#route for NL to RU not direct
route_NL_to_RU_notdirect = joblib.load('./processing/pr_inter_output/potential_routes_allNLport_to_allRUport__timeinf_nrtotport7.joblib')


min_size = min(len(sublist) for sublist in route_NL_to_RU_notdirect)
max_size = max(len(sublist) for sublist in route_NL_to_RU_notdirect)
route_hotspot_to_NL_v1 = []
for nr_port in range(min_size+1, max_size+2):
    
    routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
        route_NL_to_RU_notdirect, alltankers_adjusted, nr_port, 1,  False, oiltype = 'all', loop_type = 'country')
    route_hotspot_to_NL_v1.append(routes) 

hotspot_to_NL_w_2ports_1imo_df = []
# Note this hotspot_to_NL_w_2ports_1imo from the hotspotsToEU script
for lst in hotspot_to_NL_w_2ports_1imo:

    df = alltankers_adjusted.loc[[lst]]
    hotspot_to_NL_w_2ports_1imo_df.append(df)
hotspot_to_NL_w_2ports_1imo_merged = pd.concat(hotspot_to_NL_w_2ports_1imo_df)

more_than_2port_1iom_hotspot_to_NL = []
for lst in route_hotspot_to_NL_v1:
    route = []
    for df in lst:
        sel_row = df.iloc[1:]
        #if (sel_row['Country'].iloc[0] in hotspot_countries_):
        # if (df['Country'].iloc[1] == df['Arr_Country'].iloc[1]):
        #     next
        # else:
        route.append(sel_row)
    more_than_2port_1iom_hotspot_to_NL.append(route)
more_than_2port_1iom_hotspot_to_NL = [pd.concat(lst) for lst in more_than_2port_1iom_hotspot_to_NL if len(lst)>0]


for ind, _ in enumerate(more_than_2port_1iom_hotspot_to_NL):

    more_than_2port_1iom_hotspot_to_NL[ind]['nr_port'] = f'ports of {ind+3}'

more_than_2port_1iom_hotspot_to_NL = pd.concat(more_than_2port_1iom_hotspot_to_NL)
# plot all countries in between 
plt_cus.stackbar_w_gr_plot(more_than_2port_1iom_hotspot_to_NL, 'Country', 'nr_port',
                       "", 10,
                       "Counts", "Countries", others = True)

min_size = min(len(sublist) for sublist in route_NL_to_RU_notdirect)
max_size = max(len(sublist) for sublist in route_NL_to_RU_notdirect)
# Assign tankers status based on oil import/export
RU_NL_tankers_indirect_status = []

for lst in RU_NL_tankers:
    lst = lst.copy()  # avoid modifying original DataFrame by reference
    lst['tankers_status'] = 'no status'
    for row in range(len(lst)):
        if (lst.iloc[row]['Country'] == 'Netherlands'):


            if 'Crude Oil Tanker' in lst.iloc[row-1]['ShipType']:

                if lst.iloc[row - 1]['crude_status'] == 'Net importer':
                    lst.at[lst.index[row], 'tankers_status'] = 'empty'
                else:
                    lst.at[lst.index[row], 'tankers_status'] = 'full'
            else:

                if lst.iloc[row - 1]['refined_status'] == 'Net importer':
                    lst.at[lst.index[row], 'tankers_status'] = 'empty'
                else:
                    lst.at[lst.index[row], 'tankers_status'] = 'full'

    RU_NL_tankers_indirect_status.append(lst)
# add ing previous country and oil status in the same row where an IMO sails from NL
RU_NL_tankers_indirect_status_preinfo = []
for lst in RU_NL_tankers_indirect_status:
    lst['pre_country'] = 'nan'
    lst['pre_crude_stat'] = 'nan'
    lst['pre_refined_stat'] = 'nan'
    for row in range(len(lst)):
        if lst.iloc[row]['tankers_status'] != 'no status':
            lst.at[lst.index[row], 'pre_country'] = lst.iloc[row-1]['Country']
            lst.at[lst.index[row], 'pre_crude_stat'] = lst.iloc[row-1]['crude_status']
            lst.at[lst.index[row], 'pre_refined_stat'] = lst.iloc[row-1]['refined_status']
            
    RU_NL_tankers_indirect_status_preinfo.append(lst)
    
RU_NL_tankers_indirect_status_preinfo = pd.concat(RU_NL_tankers_indirect_status_preinfo)
# extract routes
route_NL_to_RU_notdirect_1imo = []
for nr_port in range(min_size+1, max_size+2):
    
    routes,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
        route_NL_to_RU_notdirect, RU_NL_tankers_indirect_status_preinfo, nr_port, 1,  False, oiltype = 'all', loop_type = 'country')
    route_NL_to_RU_notdirect_1imo.append(routes)
# remove unwanted routes
route_NL_to_RU_notdirect_1imo_filtered = []
for lst in route_NL_to_RU_notdirect_1imo:
    new_lst = []
    for df in lst:
        if (df.iloc[0]['Arr_Country'] in ['Netherlands', 'Russia']):
            next
        else:
            new_lst.append(df)
    route_NL_to_RU_notdirect_1imo_filtered.append(new_lst)
# extract the second trip of ports of 4
snd_trip_of_4_ports = []
for df in route_NL_to_RU_notdirect_1imo_filtered[1]:

        snd_row = df.iloc[[1]]
        snd_trip_of_4_ports.append(snd_row)
snd_trip_of_4_ports = pd.concat(snd_trip_of_4_ports)
dep_arr_port_of_4 = list(zip(snd_trip_of_4_ports['Country'], snd_trip_of_4_ports['Arr_Country']))
tanker_type = snd_trip_of_4_ports['ShipType']
count_tanker_type_portof4 = collections.Counter(tanker_type)
count_dep_arr_portof4 = collections.Counter(dep_arr_port_of_4)

# understanding where ship from Greece going to 
from_greece_to = alltankers_adjusted[alltankers_adjusted['Country'] == 'Greece']
count_from_greece_to = collections.Counter(from_greece_to['Arr_Country'])
count_shiptype_greece = collections.Counter(from_greece_to['ShipType'])
# extract imo from NL->norway/UK->RU->?
norway = []
for row in range(len(snd_trip_of_4_ports)):
    df = snd_trip_of_4_ports.iloc[[row]]
    if df['Country'].iloc[0] == 'Norway':
        norway.append(df)
norway = pd.concat(norway)
greece_trace = []
imo_greece = norway['IMO'].unique()
for imo in imo_greece:
    df = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    greece_trace.append(df)
# extract uk trip of 3
snd_trip_of_3_ports = []
for df in route_NL_to_RU_notdirect_1imo_filtered[0]:

        snd_row = df.iloc[[1]]

        if snd_row['Country'].iloc[0] == 'United Kingdom':
            snd_trip_of_3_ports.append(snd_row)
snd_trip_of_3_ports = pd.concat(snd_trip_of_3_ports)
####
# extract the previous trip before arriving into the NL
trip_bf_NL_to_RU_notdirect = []
for lst in route_NL_to_RU_notdirect_1imo_filtered:
    for df in lst:
        first_row = df.iloc[[0]]
        trip_bf_NL_to_RU_notdirect.append(first_row)
trip_bf_NL_to_RU_notdirect = pd.concat(trip_bf_NL_to_RU_notdirect)
cntry_bf_NL_RU_notdirect = collections.Counter(trip_bf_NL_to_RU_notdirect['pre_country'])

########## From another part one from countries right before NL before departing to RU with direct and the other with indirect
# Countries belongs above
# --- Prepare data for second plot ---
cntry_bf_NL_RU_notdirect_sorted = sorted(cntry_bf_NL_RU_notdirect.items(), key=lambda x: x[1], reverse=True)
top_cntry_bf_NL_RU_notdirect = cntry_bf_NL_RU_notdirect_sorted[:10]

# Sum 'Others'
others_sum = sum(v for _, v in cntry_bf_NL_RU_notdirect_sorted[10:])
top_cntry_bf_NL_RU_notdirect.append(("Others", others_sum))

countries2 = [c for c, _ in top_cntry_bf_NL_RU_notdirect]
counts2 = [v for _, v in top_cntry_bf_NL_RU_notdirect]

# Update countries
countries = ['USA' if c == 'United States of America' else c for c in countries]
countries2 = ['USA' if c == 'United States of America' else c for c in countries2]
fig, axes = plt.subplots(2, 1, figsize=(14, 12))  # 2 rows, 1 column

# --- Top subplot ---
axes[0].bar(countries, counts, color='steelblue')
axes[0].set_ylabel('Counts', fontsize=30)

axes[0].set_xticks(range(len(countries)))  # set positions for x-ticks
axes[0].set_xticklabels(countries, rotation=45, ha='right', fontsize=25)
axes[0].tick_params(axis='y', labelsize=25)

# --- Bottom subplot ---
axes[1].bar(countries2, counts2, color='steelblue')
axes[1].set_ylabel('Counts', fontsize=30)
axes[1].set_xlabel('Countries', fontsize=30)
axes[1].set_xticks(range(len(countries2)))  # positions for bottom x-ticks
axes[1].set_xticklabels(countries2, rotation=45, ha='right', fontsize=25)
axes[1].tick_params(axis='y', labelsize=25)


plt.tight_layout()
plt.show()

# Save the plot as a PDF
plt.savefig("./screenshots/cntrrightbeforeNLengotoRU.pdf", format='pdf')
#########################
plt_cus.bar_top_plot(cntry_bf_NL_RU_notdirect, 10, 
             'Top countries a IMO visited before arriving to a Dutch port and then going to direct to a Russian port')
cntry_bf_NL_RU_notdirect = sorted(cntry_bf_NL_RU_notdirect.items(), key=lambda x: x[1], reverse=True)
top_cntry_bf_NL_RU_notdirect = cntry_bf_NL_RU_notdirect[:10]

others_sum = sum(v for _, v in cntry_bf_NL_RU_notdirect[10:])
top_cntry_bf_NL_RU_notdirect.append(("Others", others_sum))

# Separate countries and counts
countries = [c for c, _ in top_cntry_bf_NL_RU_notdirect]
counts = [v for _, v in top_cntry_bf_NL_RU_notdirect]
# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(countries, counts, color='steelblue')

# Formatting
ax.set_ylabel('Counts', fontsize=30)

ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

plt.tight_layout()
plt.show()
##############
# IMO NL not dir to RU
imo_nl_nodir_ru = trip_bf_NL_to_RU_notdirect['IMO'].unique()
imo_nl_dir_ru = route_NL_to_RU_direct_merged['IMO'].unique()
with open("./processing/pr_inter_input/look_up_nl_ru_dir_lookup.pkl", "rb") as f:
    look_up_nl_ru_dir = pickle.load(f)
with open("./processing/pr_inter_input/look_up_nl_ru_not_dir_lookup.pkl", "rb") as f:
    look_up_nl_ru_not_dir = pickle.load(f)
trip_bf_NL_to_RU_notdirect['flag_state'] =  trip_bf_NL_to_RU_notdirect['IMO'].map(look_up_nl_ru_not_dir)
route_NL_to_RU_direct_merged['flag_state'] = route_NL_to_RU_direct_merged['IMO'].map(look_up_nl_ru_dir)
nat_NL_to_RU = pd.concat([trip_bf_NL_to_RU_notdirect,route_NL_to_RU_direct_merged])



plt_cus.treemap(list(nat_NL_to_RU['flag_state']), 'Nationalities of IMOs sailing from NL to RU')
count_nat = collections.Counter(nat_NL_to_RU[['flag_state']])
####################


route_NL_to_RU_notdirect_1imo_filtered = [pd.concat(lst) for lst in route_NL_to_RU_notdirect_1imo_filtered]
# add route number into each route
for ind, _ in enumerate(route_NL_to_RU_notdirect_1imo_filtered):

    route_NL_to_RU_notdirect_1imo_filtered[ind]['nr_port'] = f'ports of {ind+3}'            

# countries visited before arriving in RU
empty = []
full = []
for lst in route_NL_to_RU_notdirect_1imo_filtered:
    lst1 = []
    lst2 = []
    for df in lst: 

        if df.iloc[0]['tankers_status'] == 'empty':
            lst1.append(df)
        elif df.iloc[0]['tankers_status'] == 'full':
            lst2.append(df)
    empty.append(lst1)
    full.append(lst2)
empty = [pd.concat(lst) for lst in empty]  
full =   [pd.concat(lst) for lst in full if len(lst)>0] 



for ind, _ in enumerate(empty):

    empty[ind]['nr_port'] = f'ports of {ind+3}'

for ind, _ in enumerate(full):

    full[ind]['nr_port'] = f'ports of {ind+3}'
    
tanker_stat =  [ empty, full]
ports_visited_bf_RU_empfull = []
for lst in tanker_stat:
    ports_visited_bf_RU = []
    for sublst in lst:
        route = []
        for df in sublst:
            sel_row = df.iloc[1:]
            if (sel_row['Country'].iloc[0] in ['Netherlands', 'Russia']):
                next
            else:
                route.append(sel_row)
        if len(route)>0:
            route = pd.concat(route)
            ports_visited_bf_RU.append(route)
        else:
            next

    ports_visited_bf_RU_empfull.append(ports_visited_bf_RU)
emp_full_tankers = []   
for lst in ports_visited_bf_RU_empfull:
    for ind, _ in enumerate(lst):
        

        lst[ind]['nr_port'] = f'ports of {ind+3}'
    emp_full_tankers.append(lst)

# %% last countries visited right before arrive to RU


last_cntr_right_bf_to_NL = []
for lst in ports_visited_bf_RU_empfull:
    colect = []
    for sublst in lst:
        for df in sublst:
            last_row = df.iloc[[-1]]
            colect.append(last_row)
    last_cntr_right_bf_to_NL.append(colect)

last_cntr_right_bf_to_NL_merge = list(chain.from_iterable(last_cntr_right_bf_to_NL))
last_cntr_right_bf_to_NL_merge = pd.concat(last_cntr_right_bf_to_NL_merge)
count_last_cntry_to_RU = collections.Counter(last_cntr_right_bf_to_NL_merge['Country'])
plt_cus.bar_top_plot(count_last_cntry_to_RU, 10, 'Top countries visited right before sailing to RU ports')


# %% plot empty and full or total for countries visted before RU
emp_full_tankers = [pd.concat(lst) for lst in emp_full_tankers]
len(emp_full_tankers)
data =  emp_full_tankers[0].groupby(["Country", "nr_port"]).size().unstack(fill_value=0)
data['sum'] = data.sum(axis = 1)
data = data.sort_values(by = 'sum', ascending = False)
top = data[:10]
others_sum = data[10:].sum(axis = 0)
other_df = pd.DataFrame([others_sum], index = ['Others'])

top = pd.concat([top, other_df])


# Drop the 'sum' column for stacking
top = top.drop(columns="sum")
# Plot stacked bar
top.plot(kind="bar", stacked=True, figsize=(10, 10))

plt.title('hi', fontsize = 40, pad = 30)
plt.ylabel('counts', fontsize = 30)
plt.xlabel('countries', fontsize =30)
plt.xticks(rotation=45, ha="right",fontsize = 30 )
plt.yticks(fontsize = 30)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.show()
 

plt_cus.stackbar_w_gr_plot(emp_full_tankers[0],
                       "Top 10 countries visted by imos before reaching RU ports",
                       10, 
                       "Counts", "Countries")

plt_cus.stackbar_w_gr_plot(emp_full_tankers[1],
                       "Top 10 countries visted by imos before reaching RU ports",
                       10, 
                       "Counts", "Countries")
combine_tankes = pd.concat(emp_full_tankers)
plt_cus.stackbar_w_gr_plot(combine_tankes,
                       "Top 10 countries visted by imos before reaching RU ports",
                       10, 
                       "Counts", "Countries")


# Example: counts_empty and counts_full

for lst in range(len(emp_full_tankers)):
    for row in range(len(emp_full_tankers[lst])):
        if 'Crude' in emp_full_tankers[lst].iloc[row]['ShipType']:
            emp_full_tankers[lst].at[emp_full_tankers[lst].index[row], 'Status'] = emp_full_tankers[lst].iloc[row]['crude_status']
        else:
           emp_full_tankers[lst].at[emp_full_tankers[lst].index[row], 'Status'] = emp_full_tankers[lst].iloc[row]['refined_status'] 
empty_w_stat = emp_full_tankers[0][['Status', 'Country']]
full_w_stat = emp_full_tankers[1][['Status', 'Country']]

counts_empty = empty_w_stat.groupby(["Status", "Country"]).size().unstack(fill_value=0)
counts_full = full_w_stat.groupby(["Status", "Country"]).size().unstack(fill_value=0)

# Combine columns to ensure consistent ordering
all_countries = sorted(list(set(counts_empty.columns) | set(counts_full.columns)))
counts_empty = counts_empty.reindex(columns=all_countries, fill_value=0)
counts_full = counts_full.reindex(columns=all_countries, fill_value=0)

# Ensure correct order of x-axis categories
status_order = ["Net exporter", "Net importer"]
counts_empty = counts_empty.reindex(status_order, fill_value=0)
counts_full = counts_full.reindex(status_order, fill_value=0)

# Set bar positions
x = np.arange(len(status_order))
width = 0.4  # width of each bar

# Assign a color to each country
colors = plt.cm.tab20.colors  # up to 20 distinct colors
country_colors = {country: colors[i % len(colors)] for i, country in enumerate(all_countries)}

fig, ax = plt.subplots(figsize=(12, 6))

# Plot stacked bars for empty
bottom_empty = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x - width/2, counts_empty[country], width, bottom=bottom_empty, label=country, color=country_colors[country])
    bottom_empty += counts_empty[country]

# Plot stacked bars for full
bottom_full = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x + width/2, counts_full[country], width, bottom=bottom_full, label=country, color=country_colors[country])
    bottom_full += counts_full[country]

# Optional: add country labels on top of each segment
for i, status in enumerate(status_order):
    bottom = 0
    for country in all_countries:
        val = counts_empty.loc[status, country]
        if val > 0:
            ax.text(x[i] - width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val
    bottom = 0
    for country in all_countries:
        val = counts_full.loc[status, country]
        if val > 0:
            ax.text(x[i] + width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val

ax.set_xticks(x)
ax.set_xticklabels(status_order, fontsize=18) 
ax.set_xticklabels(status_order, fontsize=14)  # X-axis tick labels
ax.set_xlabel("Status", fontsize=20)           # X-axis label
ax.set_ylabel("Count", fontsize=20)            # Y-axis label
ax.set_title("Returned countries from a Dutch port (empty tankers vs full tankers", fontsize=18)  # Title
# Increase tick label size for both axes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()


# extract trip from NL to RU directly from the original data
RU_NL_tankers = []

for imo in alltankers_adjusted_w_status['IMO'].unique():
    poc = alltankers_adjusted_w_status[alltankers_adjusted_w_status['IMO'] == imo]
    if (poc['Country'] == 'Netherlands').any() & (poc['Arr_Country'] == 'Russia').any():
        RU_NL_tankers.append(poc)

last_cntr_right_bf_to_NL_merge_v2 = combine_tankes.loc[last_cntr_right_bf_to_NL_merge.index]

count_last_cntry_to_RU_v2 = collections.Counter(last_cntr_right_bf_to_NL_merge_v2['Country'])
plt_cus.bar_top_plot(count_last_cntry_to_RU_v2, 10, 'Top countries visited right before sailing to RU ports')
plt_cus.stackbar_w_gr_plot(last_cntr_right_bf_to_NL_merge_v2, 'Top countries visited right before sailing to RU ports', 10, 'count', 'country')

################## General A country back to the Russia, from EU directly IMPORTANT

an_Eu_to_RU = alltankers_adjusted[
    alltankers_adjusted['Country'].isin(eu_countries) &
    alltankers_adjusted['Arr_Country'].isin(['Russia'])
]
count_an_Eu_to_RU = collections.Counter(an_Eu_to_RU['Country'])
# Convert to DataFrame
df = pd.DataFrame(list(count_an_Eu_to_RU.items()), columns=["Country", "Count"])

# --- Step 1: Sort descending ---
df = df.sort_values("Count", ascending=False)

# --- Step 2: Keep top 11, rest as "Other" ---
top11 = df.head(11)
other_sum = df["Count"].iloc[11:].sum()

# Add "Other" row
final_df = pd.concat([
    top11,
    pd.DataFrame({"Country": ["Other"], "Count": [other_sum]})
])

# --- Step 3: Plot horizontal bar chart ---
plt.figure(figsize=(10,7))
plt.barh(final_df["Country"], final_df["Count"], color="skyblue", edgecolor="black")


plt.xlabel("Count", fontsize=30)
plt.ylabel("Country", fontsize=30)

# Flip y-axis so biggest is on top
plt.gca().invert_yaxis()

plt.xticks(fontsize=25)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.show()
plt.savefig('./screenshots/aEUcntrybacktoRU.pdf', format='pdf')
#################################################################################
# Coutnry bf EU then to RU IMPORTANCE

EU_RU_tankers = []

for imo in alltankers_adjusted['IMO'].unique():
    poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    if (poc['Country'].isin(eu_countries).any()) & (poc['Arr_Country'] == 'Russia').any():
        EU_RU_tankers.append(poc)
# Assign tankers status based on oil import/export

cntry_bf_NL_RU_fromallEU = []

for df in EU_RU_tankers:

    for row in range(len(df)):
        if (df.iloc[row]['Country'] in eu_countries) and (df.iloc[row]['Arr_Country'] == 'Russia'):
            cntry_bf_NL_RU_fromallEU.append(df.iloc[[row-1]])
cntry_bf_NL_RU_fromallEU = pd.concat(cntry_bf_NL_RU_fromallEU)
bf_Italy_RU = cntry_bf_NL_RU_fromallEU[cntry_bf_NL_RU_fromallEU['Arr_Country'] == 'Italy']
a_count_Italy = collections.Counter(bf_Italy_RU['Country'])
bf_Greece_RU = cntry_bf_NL_RU_fromallEU[cntry_bf_NL_RU_fromallEU['Arr_Country'] == 'Greece']
a_count_Greece = collections.Counter(bf_Greece_RU['Country'])
bf_Poland_RU =cntry_bf_NL_RU_fromallEU[cntry_bf_NL_RU_fromallEU['Arr_Country'] == 'Poland']
a_count_Poland = collections.Counter(bf_Poland_RU['Country'])

# Create DataFrames
NL1 = pd.DataFrame({"Country": countries, "Count": counts})
NL2 = pd.DataFrame({"Country": countries2, "Count": counts2})

# Merge and sum counts
ctry_bf_NL = pd.concat([NL1, NL2]) \
           .groupby("Country", as_index=False) \
           .sum()
ctry_bf_NL = ctry_bf_NL.set_index("Country")["Count"].to_dict()
ctry_bf_NL = collections.Counter(ctry_bf_NL)
ctry_bf_NL.pop("Others", None)
# Extract top 5 from each counter
top5_Italy = dict(a_count_Italy.most_common(5))
top5_Greece = dict(a_count_Greece.most_common(5))
top5_Poland = dict(a_count_Poland.most_common(5))
top5_NL = dict(ctry_bf_NL.most_common(5))
# --- Compute max value across all top5 datasets ---
max_value = max(
    max(top5_Italy.values()),
    max(top5_Greece.values()),
    max(top5_Poland.values()),
    max(top5_NL.values())
)

# --- Create subplots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

# --- Italy ---
axes[0].barh(list(top5_Italy.keys()), list(top5_Italy.values()), 
             color="skyblue", edgecolor="black")
axes[0].set_title("Italy", fontsize=40)
axes[0].set_xlabel("Count", fontsize=40)
axes[0].set_ylabel("Country", fontsize=40)
axes[0].tick_params(axis='x', labelsize=30)
axes[0].tick_params(axis='y', labelsize=35)
axes[0].set_xlim(0, max_value * 1.1)  # same scale for all

# --- Greece ---
axes[1].barh(list(top5_Greece.keys()), list(top5_Greece.values()), 
             color="skyblue", edgecolor="black")
axes[1].set_title("Greece", fontsize=40)
axes[1].set_xlabel("Count", fontsize=40)
axes[1].tick_params(axis='x', labelsize=30)
axes[1].tick_params(axis='y', labelsize=35)
axes[1].set_xlim(0, max_value * 1.1)

# --- Poland ---
axes[2].barh(list(top5_Poland.keys()), list(top5_Poland.values()), 
             color="skyblue", edgecolor="black")
axes[2].set_title("Poland", fontsize=40)
axes[2].set_xlabel("Count", fontsize=40)
axes[2].tick_params(axis='x', labelsize=30)
axes[2].tick_params(axis='y', labelsize=35)
axes[2].set_xlim(0, max_value * 1.1)

# --- Netherlands ---
axes[3].barh(list(top5_NL.keys()), list(top5_NL.values()), 
             color="skyblue", edgecolor="black")
axes[3].set_title("Netherlands", fontsize=40)
axes[3].set_xlabel("Count", fontsize=40)
axes[3].tick_params(axis='x', labelsize=30)
axes[3].tick_params(axis='y', labelsize=35)
axes[3].set_xlim(0, max_value * 1.1)

plt.tight_layout()
plt.show()
plt.savefig('./screenshots/GreItaPol.pdf', format='pdf')

# Check NOv. distribution

nov_dis = alltankers_adjusted[alltankers_adjusted['DepPort'] == 'Novorossiysk']
count_nov_dis = collections.Counter(nov_dis['Arr_Country'])
top11 = count_nov_dis.most_common(11)
countries = [c for c, _ in top11]
values = [v for _, v in top11]

# --- Rename for clarity (optional) ---
countries = ["USA" if c == "United States of America" else 
              "UK" if c == "United Kingdom" else c for c in countries]

# --- Plot ---
plt.figure(figsize=(12, 7))
plt.barh(countries, values, color='skyblue')
plt.gca().invert_yaxis()  # Largest on top

#plt.title("Top 11 Destination Countries (November–December)", fontsize=20, fontweight='bold')
plt.xlabel("Count", fontsize=35)
plt.ylabel("Country", fontsize=35)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)

plt.tight_layout()
plt.show()
plt.savefig('./screenshots/Novdistribution.pdf', format='pdf')

# belgium

Bel_dis = alltankers_adjusted[(alltankers_adjusted['Country'] == 'Belgium') & (alltankers_adjusted['Arr_Country'] == 'Netherlands')]
USA_dis = alltankers_adjusted[(alltankers_adjusted['Country'] == 'United States of America') & (alltankers_adjusted['Arr_Country'] == 'Netherlands')]
Uk_dis = alltankers_adjusted[(alltankers_adjusted['Country'] == 'United Kingdom') & (alltankers_adjusted['Arr_Country'] == 'Netherlands')]
Germany_dis = alltankers_adjusted[(alltankers_adjusted['Country'] == 'Germany') & (alltankers_adjusted['Arr_Country'] == 'Netherlands')]
Norway_dis = alltankers_adjusted[(alltankers_adjusted['Country'] == 'Norway') & (alltankers_adjusted['Arr_Country'] == 'Netherlands')]
country_dis = [Bel_dis, USA_dis, Uk_dis, Germany_dis, Norway_dis]
count_dis = []
for ctry in country_dis:
    count = collections.Counter(ctry['ShipType'])
    count_dis.append(count)

# Define categories to keep separate
keep = {'Crude Oil Tanker', 'Crude/Oil Products Tanker'}
new_counts = []
for counts in count_dis:
    merged_counts = collections.Counter()
    for ship_type, value in counts.items():
        if ship_type in keep:
            merged_counts[ship_type] += value
        else:
            merged_counts['Refined Oil Tanker'] += value
    new_counts.append(merged_counts)

titles = ["Belgium", "United States", "United Kingdom", "Germany", "Norway"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # consistent colors for categories
category_names = ['Crude Oil Tanker', 'Crude/Oil Products Tanker', 'Refined Oil Tanker']

# --- Determine common y-axis scale ---
max_value = max(max(c.values()) for c in new_counts)

# --- Create subplot grid ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

bars_for_legend = []  # store one bar per category for legend

for i, ax in enumerate(axes):
    if i < len(new_counts):
        counter = new_counts[i]
        values = [counter.get(cat, 0) for cat in category_names]

        bars = ax.bar(category_names, values, color=colors)
        ax.set_title(titles[i], fontsize=25, fontweight='bold')
        ax.set_ylim(0, max_value * 1.1)
        ax.tick_params(axis='x', labelbottom=False)  # remove x-axis tick labels
        ax.tick_params(axis='y', labelsize=30)
        # --- Add y-axis label ---
        ax.set_ylabel("Frequency", fontsize=30)  # <-- this adds the y-axis title

        # store bars for legend (only once)
        if i == 0:
            bars_for_legend = bars
    else:
        # remove unused subplot
        fig.delaxes(ax)

# --- Add a single legend at the bottom ---
fig.legend(bars_for_legend, category_names, loc='lower center', ncol=3, fontsize=25, frameon=False)

plt.tight_layout(rect=[0, 0.09, 1, 0.95])  # leave space at bottom for legend
plt.show()
plt.savefig('./screenshots/nonhotspottoNL.pdf', format='pdf')