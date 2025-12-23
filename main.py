# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 02:23:26 2025

@author: Duyen
This script shows how to run the path-searching algorithm, and test the algorithm scalability
""" 
import os
cwd = os.getcwd()
#os.chdir('D:\\Dropbox\\Duyen\\University\\Master\\Year 2\\Internship\\')
processes = os.cpu_count() - 2
import sys
import time
import networkx as nx
from Code import data_processing as pr
from Code import routefinding_v2 as rf2
from Code import tracking_time_mem as trtm
import pandas as pd
import joblib
import csv
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %% loading data
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
    
# import data

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

# %% Port selections for different regions

country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Russia', 'Netherlands']
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


updated_hotspot_countries = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Malaysia']
ports_of_hotspots = {}
for ctry in updated_hotspot_countries:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    ports_of_hotspots[ctry] = list(ctry_ports)
    
# only 4 hotspots
four_hotspot = ['China', 'India', 'Turkey', 'Kazakhstan']
port_of_4_hotspots = {}
for ctry in four_hotspot:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    port_of_4_hotspots[ctry] = list(ctry_ports)
# Select RU port

ru_country = ['Russia']
port_of_russia = pr.extract_ports_based_countries(alltankers_adjusted, ru_country)


eu_ports = pr.extract_ports_based_countries(alltankers_adjusted, eu_countries)
# Select the Dutch ports

NL_ports =  pr.extract_ports_based_countries(alltankers_adjusted, NL)
# %% Creating graph and network from the port call data
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

# %% This part can be ignored if you already know ports you are interested in

# betweeness centrality
btwcentr = nx.betweenness_centrality(Graph_whole_dataset)
# extract betweeness centrality of ports for each country
bwtcentr_ports = []
for ctr_of_int in country_of_interest:
    filter_value = {port: btwcentr[port] for port in port_of_interest[ctr_of_int]}
    # Get top 2 keys with highest values
    top_2_keys = sorted(filter_value, key=filter_value.get, reverse=True)[:2]
    bwtcentr_ports.append(top_2_keys)
bwtcentr_ports = [port for sublist in bwtcentr_ports for port in sublist]

# NOTE
# choose one of the methods to select start port and end ports for the algorithm
# If you do not set any limitations the number of IMO in the extracted routes,
# and the maximun number of total ports >5, and include cyclic routes.
# We recommend you to only select ports with high betweeness centrality to 
# reduce the runtime
# then use this code
### top 5 ports for each hot spots
# btw_ctral_hotpots_ports = []
# for ctr_of_int in ports_of_hotspots:
#     filter_value = {port: btwcentr[port] for port in ports_of_hotspots[ctr_of_int]}
#     # Get top 2 keys with highest values
#     top_ports = sorted(filter_value, key=filter_value.get, reverse=True)[:2]
#     btw_ctral_hotpots_ports.append(top_ports)
# btw_ctral_hotpots_ports = [port for sublist in btw_ctral_hotpots_ports for port in sublist]
# # extract ports of NL and RU with the highest betweeness centrality
# start_RU_port = list(set(bwtcentr_ports) & set(port_of_russia))
# end_port = list(set(bwtcentr_ports) & set(NL_ports))
# # remove RU and NL port from the port lists of highest betweeness centrality
# for port in start_RU_port:
#     bwtcentr_ports.remove(port)
# for port in end_port:
#     bwtcentr_ports.remove(port)

    

# if you set the number of IMO to 1 for each route, you can use all start ports,
# and end ports exisitng in your data
# then use this code
# extract all the values out
all_hotspots_ports = list(ports_of_hotspots.values())
all_hotspots_ports = [port for sublist in all_hotspots_ports for port in sublist]


start_RU_port = all_hotspots_ports
end_port  = eu_ports




# %% PESUDO CODE: find routes operated by a single tanker (nr_IMO = 1)
# Example:  extract routes from all hotspot ports to all EU ports

start_time = time.time()
upperbound_time = float('inf')
lowerbound_time = 0
win_time_slide = 1
iterat_time = 1
strike = 'None'
RU_to_NL = False
loop = True
loop_type = 'country'
IMO_con = True
RU_to_NL_con = False
port_of_interest = bwtcentr_ports
nr_imo = 1
tot_nr_port = 7
outputpath = f'./processing/pr_inter_output/potential_routes_allhotspot_to_allEUport__timeinf_nrtotport7.joblib'
# Result

final_route_RU_to_NL = rf2.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                  win_time_slide, iterat_time, 
                  strike, tot_nr_port, nr_imo, outputpath, Graph_whole_dataset, 
                  port_of_russia, eu_ports, port_of_interest,
                  alltankers_adjusted, ru_country, loop, loop_type, RU_to_NL, RU_to_NL_con, IMO_con)
# Function used to further filter the extracted routes into the user requirement
route_RU_int_NL_matched_imoNr,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    final_route_RU_to_NL, alltankers_adjusted, 3, 1,  False, oiltype = 'all', loop_type = 'country')




# %% PESUDO CODE: Find all routes (nr_IMO = 'all')
# Note: should not use all ports exisiting in the dataset. Select ports based on betweeness centrality as in the code above
nr_of_port = 5
iterat_time = 23
start_time = time.time()
upperbound_time = 24*7 #float('inf') - one week interval
lowerbound_time = 0
win_time_slide = 1
strike = 'None'
RU_to_NL = False
loop = False
loop_type = 'country'
IMO_con = True
RU_to_NL_con = False
port_of_interest = bwtcentr_ports
nr_imo = 'all'
outputpath = f'./processing/pr_inter_output/all_potential_routes_loop__nrRU_{len(start_RU_port)}_time1w_nrtotport_{nr_of_port}.joblib'

final_route_RU_to_NL = rf2.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                  win_time_slide, iterat_time, 
                  strike, nr_of_port, nr_imo, outputpath, Graph_whole_dataset, 
                  port_of_russia, eu_ports, port_of_interest,
                  alltankers_adjusted, ru_country, loop, loop_type, RU_to_NL, RU_to_NL_con, IMO_con)
# Function used to further filter the extracted routes into the user requirement
route_RU_int_NL_matched_imoNr,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
    final_route_RU_to_NL, alltankers_adjusted, 4, 3,  False, oiltype = 'all', loop_type = 'country')

# %% PESUDO CODE to assess the algorithm scalability across total number of ports, and with nr_IMO = 'all'
tot_nr_port = (3, 4, 5, 6, 7, 8)

# for 1 week interval
iterat_time = (25, 25, 23, 19, 15, 11)
# for 2 week interval
#iterat_time = (16,14,12,10,8,6)
# for 3 and 4 week interval
# iterat_time = (8,7,6, 5, 4, 3)
zip_iter_and_nrport = list(zip(tot_nr_port,iterat_time))
port_of_interest = bwtcentr_ports

def func(zip_iter_and_nrport):
#     for tup in zip_iter_and_nrport:
            nr_of_port, iterat_time = zip_iter_and_nrport

            start_time = time.time()
            upperbound_time = 24*7 #float('inf')
            lowerbound_time = 0
            win_time_slide = 1
            strike = 'None'
            RU_to_NL = True
            loop = False
            loop_type = 'country'
            IMO_con = True
            RU_to_NL_con = True
            port_of_interest = bwtcentr_ports
            nr_imo = 'all'
            outputpath = f'./processing/pr_inter_output/all_potential_routes_loop__nrRU_{len(start_RU_port)}_time1w_nrtotport_{nr_of_port}.joblib'
            
            
            final_route_RU_to_NL = rf2.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                              win_time_slide, iterat_time, 
                              strike, nr_of_port, nr_imo, outputpath, Graph_whole_dataset, 
                              port_of_russia, eu_ports, port_of_interest,
                              alltankers_adjusted, ru_country, RU_to_NL, RU_to_NL_con, IMO_con, loop, loop_type)
            return final_route_RU_to_NL


# Set up the runtime and memory usage
        
sys_list = []
proc_list = []
if __name__ == "__main__":
    for zipl in zip_iter_and_nrport:
        sys_log_file = f'./processing/pr_inter_output/all_sys_log_file_loop_RU_{len(start_RU_port)}_1w_port_{zipl[0]}.csv'
        proc_log_file = f'./processing/pr_inter_output/all_proc_log_file_loop_RU_{len(start_RU_port)}_1w_port_{zipl[0]}.csv'

        result, sys_log, proc_log = trtm.run_with_dual_memory_tracking(
            func,
            func_args=(zipl,),
            log_interval=0.5
        )
        sys_list.append(list(sys_log))
        proc_list.append(list(proc_log))
        # Save system memory log to CSV
        with open(sys_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "RSS Memory (MB)", "System Memory (%)"])
            writer.writerows(sys_log)
        
        # Save process memory log to CSV
        with open(proc_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Process Memory (MB)"])
            for val in proc_log:
                writer.writerow([val])


