# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:27:39 2025

@author: Duyen
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools as it
import networkx as nx


def check_nan_duplicate(df):
    #"tanker_were_to_NL_ww"
    print('The number of Nan:', df.isnull().values.sum())
    
    NaN_df = df[df.isna().any(axis=1)]
    print('Rows that contain Nan:\n', NaN_df)
    # check dublicate
    print('The number of duplicated rows:', df.duplicated().sum())
    print('Rows that contains duplicated values:\n')
    df[df.duplicated() == True]
    # if dublicated, drop
    print('Total of rows before removing duplicated:', len(df))
    df = df.drop_duplicates()
    print('Total of rows after removing duplicated:', len(df))
    return df

def check_IMO(df):
    #"tanker_were_to_NL_ww"
    # Check for the standard form of IMO
    IMO_con_check = df['IMO'].map(lambda x: 'good ' if ((len(str(x)) == 7) & (str(x).isdigit())) else  'bad').to_frame(name = 'CON')
    if 'bad' in IMO_con_check['CON']:
        print('Indices containing wrong IMO::\n')
        print(IMO_con_check['CON'].index[IMO_con_check['CON'].str.contains('bad')].tolist())
    else:
        print('All IMO meets the condition')
    
    
def validate_timestamp_sign(df): 
    #"tanker_were_to_NL_ww"
    # validate the order consistancy of arrival date, sail date, last port call sail date
    arrival_sail = (df["SAILDATE"] - df["ARRIVALDATE"])
    
    if ((arrival_sail.dt.days >= 0).all() & (not arrival_sail.isna().any())):
        print("Arrival date and sail date are in good order")
    else:
        arrival_sail = arrival_sail.to_frame(name = "diff")
        diff_neg = arrival_sail['diff'].index[arrival_sail['diff'].dt.days < 0].tolist()
        print('arrival_sail-Index with negative values:', diff_neg)
        diff_nan = arrival_sail['diff'].index[arrival_sail['diff'].isnull()].tolist()
        print('arrival_sail-Index with nan values:', diff_nan)
        
    arrival_lastPOC = (df["ARRIVALDATE"] - df["LASTPORTOFCALLSAILDATE"])
    
    if ((arrival_lastPOC.dt.days >= 0).all() & (not arrival_lastPOC.isna().any())):
        print("Arrival date and last POC sail date are in good order")
    else:
        arrival_lastPOC = arrival_lastPOC.to_frame(name = "diff")
        diff_neg = arrival_lastPOC['diff'].index[arrival_lastPOC['diff'].dt.days < 0].tolist()
        print('arrival_lastPOC-Index with negative values:', diff_neg)
        diff_nan = arrival_lastPOC['diff'].index[arrival_lastPOC['diff'].isnull()].tolist()
        print('arrival_lastPOC-Index with nan values:', diff_nan)
        
def check_duplicated_timestamp(df):
    #"tanker_were_to_NL_ww"

    # check identical arrival date, sail date, last POC sail date
    
    df['DUPLICATE_2col'] = df.apply(lambda row: row["ARRIVALDATE"] ==  row['SAILDATE'], axis = 1)
    print("summary of duplicated arrival and sail date:\n", df['DUPLICATE_2col'].value_counts())
    arrvl_sail_dup = df[df['DUPLICATE_2col'] == True]
    IMO_, count_ = np.unique(arrvl_sail_dup['IMO'], return_counts = True)
    IMO_freq_arrvl_sail_dup = pd.DataFrame(
        np.vstack((IMO_, count_)).T,
        columns=['IMO', 'Frequency']
    )
    
    df['DUPLICATE_3col'] = df.apply(lambda row: row["ARRIVALDATE"] ==  row['SAILDATE'] == row['LASTPORTOFCALLSAILDATE'], axis = 1)
    print("summary of duplicated arrival and sail date and the last sail date:\n", df['DUPLICATE_3col'].value_counts())
    arrvl_sail_lastPOC_dup = df[df['DUPLICATE_3col'] == True]
    IMO_, count_ = np.unique(arrvl_sail_lastPOC_dup['IMO'], return_counts = True)
    IMO_freq_arrvl_sail_lastPOC_dup = pd.DataFrame(
        np.vstack((IMO_, count_)).T,
        columns=['IMO', 'Frequency']
    )
    return arrvl_sail_dup, IMO_freq_arrvl_sail_dup, arrvl_sail_lastPOC_dup, IMO_freq_arrvl_sail_lastPOC_dup

def standadize_ship_type(name):
    #"tanker_were_to_NL_ww"
    # change position of words in parenthesis
    rgx = r"(.+)(\(.+\))"
    name = re.sub(rgx, lambda m: f"{m.group(2)} {m.group(1)}".strip() if m.group(2) else m.group(1), name)
    name = re.sub(r"[()]", "", name)
    rgx_2 = r"(.+?),\s*(.+)"
    match = re.match(rgx_2, name)
    
    if match:
        part1 = match.group(1).strip()
        part2 = re.sub(r'\bWaterways\b', '', match.group(2), flags=re.IGNORECASE).strip()
        name = f"{part1.title()} ({part2.title()})"

    return name

def standardize_port_name(name):
    #"tanker_were_to_NL_ww"
    # Remove parentheses content like (Egypt), (USA), etc.
    name = re.sub(r"\s*\(.*?\)", "", name)
    
    # Normalize FPSO/FSO/FLNG names
    name = re.sub(r"^(FPSO|FSO|FLNG)?\s*['\"]?([A-Za-z0-9\s\-\.]+)['\"]?\s*(FPSO|FSO|FLNG)?$", 
                  lambda m: f"{m.group(1) or m.group(3)} {m.group(2)}".strip() if m.group(1) or m.group(3) else m.group(2), 
                  name)
    
    # Replace '&' with 'and'
    name = name.replace("&", "and")
    
    # Remove extra spaces
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = re.sub(r"-", " ", name)
    name = re.sub(r"oil", "", name, flags=re.IGNORECASE)  # Remove all case versions of "oil"
    name = re.sub(r"\s+", " ", name).strip()  # Clean up extra spaces

    return name

def validate_hrinport(df):
    #"tanker_were_to_NL_ww"
    # validate HOURSINPORT
    seconds_in_day = 24*60*60
    df['DURATIONINPORT_TIME'] = df['SAILDATE'] - df['ARRIVALDATE']
    df['DURATIONINPORT_HR'] = round((df['DURATIONINPORT_TIME'].dt.days * seconds_in_day + df['DURATIONINPORT_TIME'].dt.seconds)/3600, 0).astype(int)
    df['HR_DIFF_Perc'] = abs(df['DURATIONINPORT_HR'] - df['HOURSINPORT'])
    large_diff_HRinPrt_actHr_list = df['HR_DIFF_Perc'].index[df['HR_DIFF_Perc'] > 1].tolist()
    print('the number of hoursinport that does not match with the calculated one:', len(large_diff_HRinPrt_actHr_list))
    print('These are indices that have large different between HOURSINPORT and self calculated HOURSINPORT:', large_diff_HRinPrt_actHr_list)
    return df, large_diff_HRinPrt_actHr_list


def plot_hourinport_freq_for_IMO(df, name):
    #"tanker_were_to_NL_ww"
    # plot hours in port distribution-Find extreme data
    # Histogram plot (countplot)
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x="DURATIONINPORT_HR", data=df)
    ax.set_title(name)
    ax.set_xlabel("Hours in Port")
    ax.set_ylabel("Frequency")

    # Limit the x-axis to focus on densest area (e.g., 0 to 700 or automatic based on quantiles)
    max_val = df["DURATIONINPORT_HR"].quantile(0.95)  # adjust as needed
    ax.set_xlim(0, max_val)
    # Set custom xticks from 0 to 700 in steps of 50, but only within visible range
    xticks = np.arange(0, min(701, max_val + 1), 25)
    ax.set_xticks(xticks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    # scater plot
    # calculate the frequency of hours in port
    df_freq = df['DURATIONINPORT_HR'].value_counts().reset_index(name = 'Frequency').rename(columns = {'index' : 'HOURSINPORT'} )

    plt.figure(figsize=(12, 6))
    plt.scatter(df_freq['DURATIONINPORT_HR'], df_freq['Frequency'], alpha=0.7, color='blue')
    plt.xlabel('Hours in Port')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
def IMO_freq_port_call(df):
    #"tanker_were_to_NL_ww"
    # frequency of port calls per ship-TODO: check whether the IMO with low frequency show incomplete data coverage?
    IMO_uniq, counts = np.unique(df['IMO'], return_counts = True)
    IMO_w_portcall = np.vstack((IMO_uniq, counts)).T
    # convert all unique and counts array into dataframe
    IMO_w_portcall_freq_uniq_df = pd.DataFrame(IMO_w_portcall, columns=['IMO', 'Frequency'])
    IMO_w_portcall_freq_uniq_df = IMO_w_portcall_freq_uniq_df.merge(df[['SHIPTYPE', 'IMO']], 
                                                  left_on = 'IMO',
                                                  right_on = 'IMO',
                                                  how = 'left')
    return IMO_w_portcall_freq_uniq_df


def plot_NO_ships_per_freq(df):
    #"IMO_NL_ww_df"
    # Define bins and labels
    bins = [0, 1, 10, 50, 100, 400, 700, 800, float('inf')]
    labels = ['1', '2–10', '11–50', '51–100', '101–400', '401–700', '701–800', '>800']
    
    # Create binned category
    df['range'] = pd.cut(df['Frequency'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # Count how many IMOs fall into each range
    range_counts = df['range'].value_counts().sort_index()
    
  
    # Plot
    plt.figure(figsize=(10, 6))
    range_counts.plot(kind='bar', color='skyblue', edgecolor='black', stacked = True)
    plt.title("Number of IMO Vessels by Frequency of Port Calls")
    plt.xlabel("Port Call Frequency Range")
    plt.ylabel("Number of Vessels (IMO)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    stacked_bar_shiptype = df.groupby('range')['SHIPTYPE'].value_counts().unstack('SHIPTYPE')
    stacked_bar_shiptype.plot.bar(stacked=True)
    plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
    plt.xlabel("Port Call Frequency Range")
    plt.ylabel("Number of Vessels (IMO)")
    plt.title("Number of IMO Vessels by Frequency of Port Calls with contributing of the ship types")
    
    return df
    
def plot_boxplot(df1, df2):
    #"tanker_were_to_NL_ww"
    #"IMO_NL_ww_df"
    # determine the relationship between tanker types and hour in port
    ax = sns.boxplot(data = df1, x = 'SHIPTYPE', y = 'DURATIONINPORT_HR')

    ax.tick_params(axis='x', rotation= 90)
    plt.title('Relationship between tanker types and their corresponding Hr in port')
    plt.show()


    # determine the relationship between tanker types and the number of port frequency
    #IMO_NL_ww_df
    ax_freq_type = sns.boxplot(data = df2, x = 'SHIPTYPE', y = 'Frequency')  

    ax_freq_type.tick_params(axis='x', rotation= 90)
    plt.title('Relationship between tanker types and their port call frequency')
    plt.show()







def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )



def one_oil_trans_fr_RU_to_NL(pot_imo_from_RU_to_NL, alltankers_adjusted,NL_ports, port_of_russia):
    one_oil_trans_fr_RU_to_NL = pd.DataFrame(columns = alltankers_adjusted.columns)
    for row in range(len(pot_imo_from_RU_to_NL)):
        pot_imo = alltankers_adjusted[alltankers_adjusted['IMO'] ==
                                      pot_imo_from_RU_to_NL['IMO'].iloc[row]].reset_index(drop = True)
        snd_target_port = pot_imo[(pot_imo['DepPort'] == pot_imo_from_RU_to_NL['DepPort'].iloc[row])
                                            & (pot_imo['DepDate'] == pot_imo_from_RU_to_NL['DepDate'].iloc[row])
                                            & (pot_imo['ArrDate'] == pot_imo_from_RU_to_NL['ArrDate'].iloc[row])]
        print('Here')
        if snd_target_port['ArrPort'].isin(NL_ports).any():
            one_oil_trans_fr_RU_to_NL = pd.concat([one_oil_trans_fr_RU_to_NL, snd_target_port])
        else:
            row_nr_of_target_2nd_port = snd_target_port.index[0]
            pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:len(pot_imo)]
            if ~pot_imo_from_2nd_target_port['DepPort'].isin(port_of_russia).any():
                # create network
                edges = []
                for n in range(len(pot_imo_from_2nd_target_port)):
                    info = tuple([pot_imo_from_2nd_target_port['DepPort'].iloc[n], 
                                  pot_imo_from_2nd_target_port['ArrPort'].iloc[n]])
                    edges.append(info)
                # create graph
                ## multi-direct-graph
                Graph_seg_route = nx.MultiDiGraph()
                Graph_seg_route.add_edges_from(edges)
                # remove loop
                Graph_seg_route.remove_edges_from(list(nx.selfloop_edges(Graph_seg_route)))
                nodes_degree = list(Graph_seg_route.degree())
    
                node_with_more_2_degree = [node for node, value in nodes_degree if value >=3]
                if len(node_with_more_2_degree) == 0:
                    one_oil_trans_fr_RU_to_NL = pd.concat([one_oil_trans_fr_RU_to_NL,
                                                           pot_imo_from_2nd_target_port])
    return one_oil_trans_fr_RU_to_NL

def second_oil_trans(tracking_snd_oiltransshipment_imo_list, alltankers_adjusted,scnd_in_day,low_t_time,up_t_time):
    tracking_all_nbs_next_oiltrans_connect_IMO = []
    for row in range(1): #tracking_snd_oiltransshipment_imo_list -has more row and each row is a different label
        tracking_each_nb_next_oiltrans_connect_IMO = []
        start_IMO_v1 = {} # start from a specific RU port. Expect a dict of dict. with the
        # first layer contain RU port names and its connected IMO info in general.
        # The secondlayer key: nb name and its attributes
        cons_IMO_v1 = {} # the next second port met conditions (no EU port) and contain a connected IMO
        cons_IMO_nr_v1 = [] # contain a connected IMO
        # loop through all neighbours of Novorossiysk
        

        # extract route from a RU port to its neighbout
        route_from_RUport_to_its_nb = tracking_snd_oiltransshipment_imo_list[row].iloc[1:2]
        arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
        # extract all IMO available at the arrival port of the first trip from RU
        diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
        # for each nb, calculate time gap between IMO from RU to 2nd port and 
        # the IMO available at the 2nd port
        
        for row in range(len(route_from_RUport_to_its_nb)):
            
            # arrive time of IMO travel from RU to its nb
            arr_time_of_IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row,
                                                  route_from_RUport_to_its_nb.columns.get_loc('ArrDate')]
            arr_port_per_row = route_from_RUport_to_its_nb.iloc[row,
                                                  route_from_RUport_to_its_nb.columns.get_loc('ArrPort')]
            IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[0:2]
            test_test_ = pd.DataFrame(columns = IMO_fr_RU_to_2ndPort.columns)
            # loop through all IMO availabel in a nb
            for row_dep_port in range(len(diff_IMO_at_2ndPort)):
                # departure time of an IMO
                dep_time_of_IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                                  diff_IMO_at_2ndPort.columns.get_loc('DepDate')]
                dep_port_per_row = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                                  diff_IMO_at_2ndPort.columns.get_loc('DepPort')]
                IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port:
                                                  row_dep_port+1]
       
                # time different between IMO from RU and IMO availabe at its nb
                time_gap = dep_time_of_IMO_avai_at_2ndPort - arr_time_of_IMO_fr_RU_to_2ndPort
                time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                # base on threshold assump to take for oil transshipment. if met 
                # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= low_t_time) & (abs(time_gap_hr) < up_t_time)):
                    #print(f'time gap after the condition {time_gap_hr}')


                    test_test_ = pd.concat([tracking_snd_oiltransshipment_imo_list[row],
                                            IMO_avai_at_2ndPort])
                    if len(test_test_) != 0:
                        
                        tracking_each_nb_next_oiltrans_connect_IMO.append(test_test_)
                    index = 1
                    # check whether the key(nb name) in the dict, not create new. 
                    # Otherwise, append 
                    if dep_port_per_row not in list(cons_IMO_v1.keys()):
                        cons_IMO_v1[dep_port_per_row] =  tuple([[
                            index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                    'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                    'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                         'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                         'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                        IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                        cons_IMO_nr_v1.append(IMO)
                    else:
                        index = len(cons_IMO_v1[dep_port_per_row]) + 1  
                        cons_IMO_v1[dep_port_per_row] += tuple([[
                            index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                    'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                    'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                         'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                         'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                        IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                        cons_IMO_nr_v1.append(IMO)
                else:
                    next

        # add main key of the dictionary, the key is a RU port name
        start_IMO_v1[arr_port_per_row] = cons_IMO_v1
    tracking_all_nbs_next_oiltrans_connect_IMO.append(tracking_each_nb_next_oiltrans_connect_IMO)
    return tracking_all_nbs_next_oiltrans_connect_IMO
    