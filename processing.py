# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:22:15 2025

@author: Duyen
"""
from itertools import islice
import sys
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import itertools
from datetime import datetime, timedelta
import networkx as nx
import csv
import re
from Code import data_processing as pr
from Code import data_preprocessing as pp
import numpy as np
import pandas as pd
import os
os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
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
alltankers = pd.read_csv(
    './preprocessing/inter_input/All Port calls - NL & RU.csv')

# select data from NL and RU ww only
alltankers = alltankers[alltankers['PATH'].isin(
    ['Tankers were to NL (worldwide)', 'Tankers were to RU (worldwide)'])]
alltankers = alltankers[['IMO', 'SHIPTYPE',
                         'COUNTRY ', 'PORTNAME', 'ARRIVALDATE', 'SAILDATE']]
alltankers = alltankers.rename(columns={'SAILDATE': 'DEPDATE'})
portname = alltankers['PORTNAME'].drop_duplicates()
# remove dublicate
alltankers = alltankers.drop_duplicates()
# standadize ship name
alltankers['SHIPTYPE'] = alltankers['SHIPTYPE'].map(
    lambda x: pp.standadize_ship_type(x))
# convert columns to the right format
alltankers['ARRIVALDATE'] = alltankers['ARRIVALDATE'].astype('datetime64[ns]')
alltankers['DEPDATE'] = alltankers['DEPDATE'].astype('datetime64[ns]')
# calculate time a vessel spent in a port for each POC
seconds_in_day = 24*60*60
alltankers['TIMEINPORT'] = alltankers['DEPDATE'] - alltankers['ARRIVALDATE']
# calculate time a vessel travel from one port to another port

alltankers_adjusted = pd.DataFrame()
for imo in alltankers['IMO'].unique():
    a_imo = alltankers[alltankers['IMO'] == imo]
    Port2Port = pd.DataFrame()
    Port2Port['IMO'] = a_imo['IMO']
    Port2Port['DepPort'] = a_imo['PORTNAME']
    Port2Port['ArrPort'] = a_imo['PORTNAME'].shift(-1)
    Port2Port['DepDate'] = a_imo['DEPDATE']
    Port2Port['ArrDate'] = a_imo['ARRIVALDATE'].shift(-1)
    Port2Port['ShipType'] = a_imo['SHIPTYPE']
    Port2Port['Country'] = a_imo['COUNTRY ']
    Port2Port['TravelTime'] = abs(Port2Port['DepDate'] - Port2Port['ArrDate'])
    Port2Port['BerthTime'] = Port2Port['DepDate'].shift(
        -1) - Port2Port['ArrDate']
    alltankers_adjusted = pd.concat([alltankers_adjusted, Port2Port])

# remove row that contain Nan
alltankers_adjusted = alltankers_adjusted.dropna(subset=['DepPort', 'ArrPort'])
# # sort values to depature date
# alltankers_adjusted = alltankers_adjusted.sort_values(by = 'DepDate')
# # Save to CSV
# alltankers_adjusted_ch = alltankers_adjusted
# alltankers_adjusted_ch = alltankers_adjusted_ch.rename(columns={'DepPort': 'Source', 'ArrPort':'Target'})
# alltankers_adjusted_ch.to_csv('./preprocessing/pp_inter_ouput/alltanker.csv', index=False)
# # Save to CSV
# alltankers_adjusted_selec = alltankers_adjusted[['IMO', 'DepPort']]

# alltankers_adjusted_selec.columns = ['IMO', 'ID']
# alltankers_adjusted_selec['Name'] = alltankers_adjusted_selec['ID']
# alltankers_adjusted_selec = alltankers_adjusted_selec.drop_duplicates(subset='ID')
# alltankers_adjusted_selec.to_csv('./preprocessing/pp_inter_ouput/node.csv', index=False)

# # standadize port name
alltankers_adjusted['DepPort'] = alltankers_adjusted['DepPort'].map(lambda x:
                                                                    pp.standardize_port_name(x))
alltankers_adjusted['ArrPort'] = alltankers_adjusted['ArrPort'].map(lambda x:
                                                                    pp.standardize_port_name(x))
port_itscountry = alltankers_adjusted[['DepPort', 'Country']]
port_itscountry = port_itscountry.drop_duplicates()
alltankers_adjusted = pd.merge(
    alltankers_adjusted, port_itscountry, left_on='ArrPort', right_on='DepPort')
alltankers_adjusted = alltankers_adjusted.rename(columns={
    'Country_x': 'Country', 'Country_y': 'Arr_Country', 'DepPort_x': 'DepPort'})
alltankers_adjusted = alltankers_adjusted.drop('DepPort_y', axis=1)


# %% Port selections for different regions
# select refinery hubs
ports_of_interest = [
    # India
    "Sikka", "Mumbai", "Paradip", "Deendayal", "Chennai", "Mundra", "Haldia", "Kattupalli", "Jawaharlal Nehru Port",

    # China
    "Qinzhou", "Zhoushan", "Zhanjiang", "Dongshan", "Tianjin", "Qingdao", "Ningbo", "Dongjiakou", "Caofeidian",
    "Lianyungang", "Shidao", "Dalian", "Yangpu", "Yantai", "Huizhou", "Shekou",

    # Türkiye (Turkey)
    "Marmara Ereglisi Terminals", "Ceyhan", "Mersin", "Aliaga", "Dortyol", "Diliskelesi", "Yarimca", "Istanbul", "Yalova",

    # United Arab Emirates (UAE)
    "Jebel Ali", "Fujairah", "Ruwais", "Sharjah", "Zirku Island", "Das Island", "Port Rashid",

    # Singapore
    "Singapore",

    # Malaysia
    "Pengerang Terminal", "Tanjung Pelepas", "Port Dickson", "Johor", "Sungai Udang", "Port Klang", "Tanjung Uban", "Malacca",

    # Indonesia
    "Balongan", "Dumai", "Tanjung Intan", "Tanjung Balai Karimun",

    # Vietnam
    "Dung Quat", "Van Phong Bay", "FPSO 'Thai Binh-VN'",

    # Saudi Arabia
    "Rabigh", "Jeddah", "King Fahd Industrial Port (Yanbu)", "Ras Tanura", "Jubail", "Jizan", "Ras Al Khafji",

    # South Africa
    "Richards Bay", "Cape Town", "Durban", "Saldanha Bay"
]
country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Russia']
ports_by_country = {}
for ctry in country_of_interest:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    ports_by_country[ctry] = list(ctry_ports)

# Select RU port

ru_country = ['Russia']
port_of_russia = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in ru_country:
        ruport = alltankers_adjusted.iloc[nr, 1]
        port_of_russia.append(ruport)
    else:
        next
port_of_russia = list(set(port_of_russia))
len(port_of_russia)
# Select EU countries
eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]
# Select NL country
NL = ['Netherlands']
# Select EU ports within EU countries

eu_ports = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in eu_countries:
        euport = alltankers_adjusted.iloc[nr, 1]
        eu_ports.append(euport)
    else:
        next
eu_ports = list(set(eu_ports))
# Select the Dutch ports

NL_ports = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in NL:
        NLport = alltankers_adjusted.iloc[nr, 1]
        NL_ports.append(NLport)
    else:
        next
        
NL_ports = list(set(NL_ports))
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
## direct-graph
direct_graph = nx.DiGraph()
direct_graph.add_edges_from(network_edges)
# Create all combination of RU and NL ports
comb_ru_nl_ports = list(itertools.product(port_of_russia, NL_ports))
# extract betweeness centrality of ports for each country
port_w_high_bwtcentr = []
for ctr_of_int in country_of_interest:
    filter_value = {port: btwcentr[port] for port in ports_by_country[ctr_of_int]}
    # Get top 2 keys with highest values
    top_2_keys = sorted(filter_value, key=filter_value.get, reverse=True)[:2]
    port_w_high_bwtcentr.append(top_2_keys)
port_w_high_bwtcentr = [port for sublist in port_w_high_bwtcentr for port in sublist]

RUport_w_high_bwtcentr = ['Novorossiysk','Ust Luga']
port_w_high_bwtcentr.remove('Novorossiysk')
port_w_high_bwtcentr.remove('Ust Luga')


# define selfloop edge and remove it
# define and remove nodes with only selfloop
Graph_whole_dataset.remove_edges_from(
    list(nx.selfloop_edges(Graph_whole_dataset)))
self_loops = list(nx.selfloop_edges(Graph_whole_dataset, keys=True))

# extract route from RU for certain IMO
# create a lookup table for ships that in Eu ports
IMO_in_EU = alltankers_adjusted[alltankers_adjusted['Country'].isin(
    eu_countries)]
# create a lookup table for ships that were to NL
IMO_in_NL = alltankers_adjusted[alltankers_adjusted['Country'].isin(NL)]

# threshold time gap in hours
up_t_time = 120 #float('inf')
low_t_time = 60
scnd_in_day = 1*24*60*60
# extract 1 hop
# aim of this task is to find a connection trip at the nb port of the first trip
# extract edges from st.Peterburg and Novorossiysk
# neighbors_Nov = list(Graph_whole_dataset.neighbors('Novorossiysk'))
# filter_neighbors_Nov = [n for n in neighbors_Nov if n not in eu_ports]
# out_edge_node_neighbour = []
# extract only neighbours that belongs to the routes going from Novorossiysk
# set iteration time
m = 4
n = 0
edges_Nov = list(set(list(Graph_whole_dataset.out_edges('Novorossiysk',))))
start_IMO = {}  # start from a specific RU port. Expect a dict of dict. with the
# first layer contain RU port names and its connected IMO info in general.
# The secondlayer key: nb name and its attributes
# the next second port met conditions (no EU port) and contain a connected IMO
cons_IMO = {}
cons_IMO_nr = []  # contain a connected IMO
# loop through all neighbours of Novorossiysk
track_route_fr_RU_to_2ndPort_and_connected_IMO = []
for edge in edges_Nov:
    # extract route from a RU port to its neighbout
    start_RU_port = edge[0]
    route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_RU_port]) &
                                                      alltankers_adjusted['ArrPort'].isin([edge[1]])]
    arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
    # extract all IMO available at the arrival port of the first trip from RU
    diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
        arr_port)]
    # for each nb, calculate time gap between IMO from RU to 2nd port and
    # the IMO available at the 2nd port

    track_route_fr_RU_to_2ndPort_and_connected_IMO, start_IMO, cons_IMO, cons_IMO_nr = pr.potential_IMO_at_shared_port(
        track_route_fr_RU_to_2ndPort_and_connected_IMO,
        route_from_RUport_to_its_nb, start_RU_port,
        diff_IMO_at_2ndPort,
        scnd_in_day, low_t_time, up_t_time, start_IMO, cons_IMO, cons_IMO_nr)

n = n+1

a = alltankers_adjusted[alltankers_adjusted['IMO'] == 9436006]
while n < m:
    for node1, value1 in start_IMO.items():
        to_delete = []
        for node2, value2 in value1.items():
            if (node2 in port_of_russia) | (node2 in eu_ports):
                to_delete.append(node2)
        for key in to_delete:
            del value1[key]

    tracking_snd_oiltransshipment_imo_list = []
    for row in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
        each_nb_route = track_route_fr_RU_to_2ndPort_and_connected_IMO[row]
        arr_pre_port = each_nb_route.iloc[len(each_nb_route)-2]['ArrPort']
        arr_next_port = each_nb_route.iloc[
            len(each_nb_route)-1]['ArrPort']
        if (arr_pre_port not in eu_ports) and (arr_pre_port not in port_of_russia):
            if (arr_next_port not in eu_ports) and (arr_next_port not in port_of_russia):
                tracking_snd_oiltransshipment_imo_list.append(
                    track_route_fr_RU_to_2ndPort_and_connected_IMO[row])

    # just for now, with IMO that has a selfloop will be deleted. Should be in the preprocessing by combining the selfloop..
    tracking_filtered_snd_oiltransshipment_imo_list = []
    for row in range(len(tracking_snd_oiltransshipment_imo_list)):
        each_nb_route = tracking_snd_oiltransshipment_imo_list[row]
        if each_nb_route.iloc[len(each_nb_route)-1]['ArrPort'] != each_nb_route.iloc[len(each_nb_route)-1]['DepPort']:
            tracking_filtered_snd_oiltransshipment_imo_list.append(
                each_nb_route)

    start_IMO_v2 = {}  # start from a specific RU port. Expect a dict of dict. with the
    # first layer contain RU port names and its connected IMO info in general.
    # The secondlayer key: nb name and its attributes
    # the next second port met conditions (no EU port) and contain a connected IMO
    cons_IMO_v2 = {}
    cons_IMO_nr_v2 = []  # contain a connected IMO

    tracking_all_nbs_next_oiltrans_connect_IMO = []

    # tracking_snd_oiltransshipment_imo_list -has more row and each row is a different label
    for row_in_imo_list in range(len(tracking_filtered_snd_oiltransshipment_imo_list)):

        # loop through all neighbours of Novorossiysk

        # extract route from a RU port to its neighbout
        route_from_RUport_to_its_nb = tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list].iloc[[
            -1]]
        arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
        # extract all IMO available at the arrival port of the first trip from RU
        diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
            arr_port)]
        # for each nb, calculate time gap between IMO from RU to 2nd port and
        # the IMO available at the 2nd port
        tracking_all_nbs_next_oiltrans_connect_IMO, start_IMO_v1, cons_IMO_v1, cons_IMO_nr_v1 = pr.potential_IMO_at_cons_shared_port(
            tracking_filtered_snd_oiltransshipment_imo_list,
            row_in_imo_list,
            tracking_all_nbs_next_oiltrans_connect_IMO,
            route_from_RUport_to_its_nb, start_RU_port,
            diff_IMO_at_2ndPort,
            scnd_in_day, low_t_time, up_t_time, start_IMO_v2, cons_IMO_v2, cons_IMO_nr_v2)
    # update
    start_IMO = start_IMO_v1
    cons_IMO = cons_IMO_v1
    cons_IMO_nr = cons_IMO_nr_v1
    track_route_fr_RU_to_2ndPort_and_connected_IMO = []
    track_route_fr_RU_to_2ndPort_and_connected_IMO = tracking_all_nbs_next_oiltrans_connect_IMO
    n = n+1

# %% Phase 2: determine IMO in NL, its route from the transit port to NL, and
# their full routes from RU to NL
# unique set of connected IMO considering from all nbs
cons_IMO_nr_uniq = set(cons_IMO_nr)

last_IMO_in_seq = []
for lst in track_route_fr_RU_to_2ndPort_and_connected_IMO:
    last_IMO_in_seq.append(lst['IMO'].iloc[-1])

last_IMO_in_seq = set(last_IMO_in_seq)

# identify which connected IMO were to the NL
cons_IMO_were_to_NL = last_IMO_in_seq.intersection(IMO_in_NL['IMO'].unique())


# loop through a dict of connected IMO at nb nodes of the first RU trip. Only
# select connected IMO that were to NL
snd_IMO_were_to_NL_df = pd.DataFrame()

for key, value in cons_IMO.items():

    for ind, timestamp in value:
        IMO_snd_port = timestamp['IMO']
        print(IMO_snd_port)
        if IMO_snd_port in cons_IMO_were_to_NL:
            each_snd_IMO_were_to_NL_df = pd.DataFrame([[IMO_snd_port, key,
                                                        timestamp['DepDate'],
                                                        timestamp['ArrDate']
                                                        ]])

            snd_IMO_were_to_NL_df = pd.concat(
                [snd_IMO_were_to_NL_df, each_snd_IMO_were_to_NL_df])
snd_IMO_were_to_NL_df.columns = ['IMO', 'DepPort', 'DepDate', 'ArrDate']
# remove IMO or 2nd ports in Eu or RU ports
snd_IMO_were_to_NL_df = snd_IMO_were_to_NL_df[~snd_IMO_were_to_NL_df['DepPort'].isin(
    eu_ports) & ~snd_IMO_were_to_NL_df['DepPort'].isin(port_of_russia)]  # add columns-depPort
# write a function that select all trips of connected IMO from 2nd port (nbs)
# to NL
# a df contain potential IMOs and its sequence travel from RU to NL port
pot_imo_from_RU_to_NL = pd.DataFrame(columns=snd_IMO_were_to_NL_df.columns)
for row in range(len(snd_IMO_were_to_NL_df)):
    imo_from_2ndport_to_NL = alltankers_adjusted[
        alltankers_adjusted['IMO'] == snd_IMO_were_to_NL_df['IMO'].iloc[row]]
    time_imo_at_NL_ports = imo_from_2ndport_to_NL[imo_from_2ndport_to_NL['DepPort'].isin(
        NL_ports)]
    time_imo_at_NL_ports = time_imo_at_NL_ports['ArrDate']
    if any(snd_IMO_were_to_NL_df['DepDate'].iloc[row] < time_imo_at_NL_ports):
        pot_imo_from_RU_to_NL = pd.concat([pot_imo_from_RU_to_NL,
                                           snd_IMO_were_to_NL_df.iloc[row: row+1]])
    else:
        next
# extract only routes that directly travek from 2nd port to NL or
# stop by more ports before arriving in NL but no more oil transhipment-the
# same ship from 2nd port to the Netherlands
# A condition for a ship that stop at more than one ports, should not visit RU
# again or visit a port included in the route more than twice

# select one hop
# select more hops but check conditions
# check in the dataframe from the timestamp at the 2nd ports to the time it
# reaches the Netherlands for the first time

# filter route from 2nd port to NL, spatial consistency (not go back to RU or
# revisit a node twice) and temporal consistency ( time arrive in NL is later
# than time leave the second port)
one_oil_trans_fr_RU_to_NL = []
for row in range(len(pot_imo_from_RU_to_NL)):
    # extract only sequence of a potential IMO
    pot_imo = alltankers_adjusted[alltankers_adjusted['IMO'] ==
                                  pot_imo_from_RU_to_NL['IMO'].iloc[row]].reset_index(drop=True)
    # locate index of oil transshipment at the 2nd port based on deptime,
    # depport, arridate
    snd_target_port = pot_imo[(pot_imo['DepPort'] == pot_imo_from_RU_to_NL['DepPort'].iloc[row])
                              & (pot_imo['DepDate'] == pot_imo_from_RU_to_NL['DepDate'].iloc[row])
                              & (pot_imo['ArrDate'] == pot_imo_from_RU_to_NL['ArrDate'].iloc[row])]
    # if it is a direct route to NL save, else only extract trip segment from
    # the moment of potential oil transshipment to the first time it arrives
    # in the NL
    one_oil_trans = pd.DataFrame(columns=alltankers_adjusted.columns)
    if snd_target_port['ArrPort'].isin(NL_ports).any():
        one_oil_trans = pd.concat([one_oil_trans, snd_target_port])
    else:
        row_nr_of_target_2nd_port = snd_target_port.index[0]

        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:len(
            pot_imo)]
        first_nl_port = pot_imo_from_2nd_target_port[
            pot_imo_from_2nd_target_port['DepPort'].isin(NL_ports)].index[0]
        # potential IMO from RU to NL with sequence of the snd port to the first NL port
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:first_nl_port]
        # if there is no RU port in that sequence then create a network
        if ~pot_imo_from_2nd_target_port['DepPort'].isin(port_of_russia).any():
            # create network
            edges = []
            for n in range(len(pot_imo_from_2nd_target_port)):
                info = tuple([pot_imo_from_2nd_target_port['DepPort'].iloc[n],
                              pot_imo_from_2nd_target_port['ArrPort'].iloc[n]])
                edges.append(info)
            # create graph
            # multi-direct-graph
            Graph_seg_route = nx.MultiDiGraph()
            Graph_seg_route.add_edges_from(edges)
            # remove loop
            Graph_seg_route.remove_edges_from(
                list(nx.selfloop_edges(Graph_seg_route)))
            nodes_degree = list(Graph_seg_route.degree())
            # check degree for each nodes in the trip segment
            node_with_more_2_degree = [node for node,
                                       value in nodes_degree if value >= 3]
            # if there are nodes with > 2 degree, the route is invalid. Otherwise, append the route
            if len(node_with_more_2_degree) == 0:
                one_oil_trans = pd.concat([one_oil_trans,
                                           pot_imo_from_2nd_target_port])
    if len(one_oil_trans) != 0:
        one_oil_trans_fr_RU_to_NL.append(one_oil_trans)

temp_one_oil_trans_fr_RU_to_NL = pd.concat(
    one_oil_trans_fr_RU_to_NL, ignore_index=True)
# retrieve route from RU to the 2nd port and make a connection with the selected IMO
# available at the second port
route_fr_RU_2ndPort_fn = []

imo_unique_fr_RU_to_NL = temp_one_oil_trans_fr_RU_to_NL['IMO'].unique()


for i in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
    seg_df = track_route_fr_RU_to_2ndPort_and_connected_IMO[i]
    if seg_df['IMO'].iloc[-1] in imo_unique_fr_RU_to_NL:
        route_fr_RU_2ndPort_fn.append(seg_df)


# Final combined list
combined_routes = []

for df2 in route_fr_RU_2ndPort_fn:
    for df1 in one_oil_trans_fr_RU_to_NL:
        if df2.iloc[-1].equals(df1.iloc[0]):
            # Drop first row of df1 to avoid duplicate
            merged = pd.concat([df2, df1.iloc[1:]], ignore_index=True)
            combined_routes.append(merged)
print("--- %s seconds ---" % (time.time() - start_time))

# save list of dataframe
with open('./processing/pr_inter_input/onetransit_routes.pkl', 'wb') as outp:
    pickle.dump(combined_routes, outp, pickle.HIGHEST_PROTOCOL)

combined_routes_ = []
for i in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
    seg_df = track_route_fr_RU_to_2ndPort_and_connected_IMO[i]
    for j in range(len(one_oil_trans_fr_RU_to_NL)):
        seg_filtered = one_oil_trans_fr_RU_to_NL[j]
        if (seg_df['IMO'].iloc[-1] == seg_filtered['IMO'].iloc[-1]):
            complete_route_ = pd.concat(
                [track_route_fr_RU_to_2ndPort_and_connected_IMO[i], one_oil_trans_fr_RU_to_NL[j]])
            combined_routes_.append(complete_route_)

combined_routes_ = [lst.drop_duplicates() for lst in combined_routes_]
# save list of dataframe
with open('./processing/pr_inter_input/onetransit_allroutes.pkl', 'wb') as outp_:
    pickle.dump(combined_routes_, outp_, pickle.HIGHEST_PROTOCOL)
del combined_routes_


# %% Phase 3: Analyse
# load data
with open('./processing/pr_inter_input/onetransit_routes.pkl', 'rb') as inp:
    combined_routes_notall = pickle.load(inp)

remove_shiptype = ['Asphalt/Bitumen Tanker', 'Oil Bunkering Tanker', 'Shuttle Tanker']
# remove routes that are the same
onetransit_route_frRU_toNL = [combined_routes_notall[x] for x, _ in enumerate(combined_routes_notall)
                              if combined_routes_notall[x].equals(combined_routes_notall[x-1]) is False]
## remove df not contain hotspot port
onetransit_fshiptype_fhotpot = [ onetransit_route_frRU_toNL[x] for x, _ in enumerate(onetransit_route_frRU_toNL)
    if (onetransit_route_frRU_toNL[x]['DepPort'].isin(port_w_high_bwtcentr)).any()
        and (~onetransit_route_frRU_toNL[x]['ShipType'].isin(remove_shiptype)).all()]
            
onetrans_dir_RU_NL = [
    df for df in onetransit_route_frRU_toNL if df.shape[0] < 3]

transit_country = [df.iloc[-1]['Country'] for df in onetrans_dir_RU_NL]
st_arr_path = [df.iloc[[-1]] for df in onetrans_dir_RU_NL]
st_arr_path_df = pd.concat(st_arr_path, ignore_index=True)
st_dep_path = [df.iloc[[0]] for df in onetrans_dir_RU_NL]
st_dep_path_df = pd.concat(st_dep_path, ignore_index=True)
st_dep_path_df['ArrCountry'] = st_arr_path_df['Country']
# remove duplicate
st_dep_path_df = st_dep_path_df.drop_duplicates()
# extract path from the second port to the NL
snd_dep_path = [df.iloc[[-1]] for df in onetrans_dir_RU_NL]
snd_dep_path_df = pd.concat(snd_dep_path, ignore_index=True)
snd_dep_path_df = snd_dep_path_df.drop_duplicates()


freq_IMO_RUport_to_2ndport = st_dep_path_df['IMO'].value_counts().reset_index()
freq_IMO_2ndport_to_NLport = snd_dep_path_df['IMO'].value_counts(
).reset_index()

# frequency occurance of each IMO from RU to hotspots spots

path_frq_RU_to_hotspot = pr.path_freq_of_a_IMO_fr_A_to_B(
    freq_IMO_RUport_to_2ndport,
    alltankers_adjusted)
print(path_frq_RU_to_hotspot.sum())
path_frq_2ndhotspot_to_NL = pr.path_freq_of_a_IMO_fr_B_to_NL(freq_IMO_2ndport_to_NLport,
                                  alltankers_adjusted)
print(path_frq_2ndhotspot_to_NL.sum())
# sum the frequency of each row
path_frq_2ndhotspot_to_NL['Sum'] = path_frq_2ndhotspot_to_NL.loc[:,
                                                                 'Ind_NL':'UAE_NL'].sum(axis=1)
sum(path_frq_2ndhotspot_to_NL['Sum'])
# percentage of IMO that revisited hotspots from RU more than >3 time
perc_IMO_revisit_hotsportfromRU = (len(
    path_frq_RU_to_hotspot[path_frq_RU_to_hotspot['Sum'] >= 2])/len(path_frq_RU_to_hotspot))*100
# percentage of IMO that were in one of the hotspot and to NL
perc_IMO_inhotspot_to_NL = (len(
    path_frq_2ndhotspot_to_NL[path_frq_2ndhotspot_to_NL['Sum'] >= 1])/len(path_frq_2ndhotspot_to_NL))*100
# IMO from RU to 2nd nonEU port vs IMO from 2nd nonEU port to NL
# number of IMO carry the whole trip to the NL
same_IMO_RU_dir_NL = []
for row in onetrans_dir_RU_NL:
    if row['IMO'][0] == row['IMO'][1]:
        same_IMO_RU_dir_NL.append(row)

# %% Visualization
# Combine all unique ShipTypes from both DataFrames
# 1. Combine all shiptypes
all_shiptypes = sorted(
    set(st_dep_path_df['ShipType'].unique()) |
    set(snd_dep_path_df['ShipType'].unique())
)

# 2. Assign consistent colors
color_palette = sns.color_palette("tab10", n_colors=len(all_shiptypes))
color_map = dict(zip(all_shiptypes, color_palette))

# 3. Group data and align columns
st_gr = st_dep_path_df.groupby('ArrCountry')[
    'ShipType'].value_counts().unstack(fill_value=0)
snd_gr = snd_dep_path_df.groupby(
    'Country')['ShipType'].value_counts().unstack(fill_value=0)

st_gr = st_gr.reindex(columns=all_shiptypes, fill_value=0)
snd_gr = snd_gr.reindex(columns=all_shiptypes, fill_value=0)

# 4. Create a common x-axis of countries
all_countries = sorted(set(st_gr.index) | set(snd_gr.index))
st_gr = st_gr.reindex(index=all_countries, fill_value=0)
snd_gr = snd_gr.reindex(index=all_countries, fill_value=0)

# 5. Plot
fig, ax = plt.subplots(figsize=(16, 6))

x = np.arange(len(all_countries))
bar_width = 0.35

# Stacked bars


def plot_stacked_bars(data, base_x, hatch=None, label_prefix=''):
    bottom = np.zeros(len(data))
    for shiptype in all_shiptypes:
        values = data[shiptype].values
        bars = ax.bar(base_x, values, bottom=bottom, width=bar_width,
                      color=color_map[shiptype], label=label_prefix + shiptype if hatch else shiptype)
        if hatch:
            for bar in bars:
                bar.set_hatch(hatch)
        bottom += values


# Plot st_ and snd_ side-by-side
plot_stacked_bars(st_gr, x - bar_width/2, hatch=None,
                  label_prefix='')         # Solid
plot_stacked_bars(snd_gr, x + bar_width/2, hatch='///',
                  label_prefix='')       # Hatched

# Legend (deduplicated)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(
    0.5, -0.25), loc='lower center', ncol=3, frameon=False)
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.5),  # (x, y) relative to plot box
    ncol=3,
    frameon=False
)
# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(all_countries, rotation=90)
ax.set_xlabel("Arrival Countries")
ax.set_ylabel("Number of Vessels")
ax.set_title(
    "Contribution of Ship Types in Oil Transport\n(1st vs 2nd Departure Paths)")

plt.tight_layout()
plt.show()


a = alltankers_adjusted[alltankers_adjusted['IMO'] == 9321847]
