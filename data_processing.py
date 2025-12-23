# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:44:55 2025

@author: Duyen
This script contains all functions required for constructing the path searching algorithm and analyzing patterns.
"""
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
# %%

def freq_of_port_seq(route_RU_int_NL_matched_imoNr):
    """Get frequency of each unique route on port level
    Parameters
    ----------
    route_RU_int_NL_matched_imoNr : List of list
        Contain lists of extracted routes encoded in row indices. These row
        indices originate from the prepocessed port call data.

    Returns
    -------
    port_sequence : dict
        A dictionary of unique routes on port level and their frequencies
        """
    port_sequence = {}
    for df in route_RU_int_NL_matched_imoNr:
    
        port_list = df['DepPort'].tolist()
        port_list.append(df.iloc[-1]['ArrPort'])
        port_list = tuple(port_list)
        if port_list not in list(port_sequence.keys()):
            port_sequence[port_list] =  1
        else:
            port_sequence[port_list] = port_sequence.get(port_list) + 1
    return port_sequence

def freq_of_country_seq(route_RU_int_NL_matched_imoNr):
    """Get frequency of each unique route on country level
    Parameters
    ----------
    route_RU_int_NL_matched_imoNr : list
        Contain a list of lists of extracted routes encoded in row indices. These row
        indices originate from the prepocessed port call data.

    Returns
    -------
    port_sequence : dict
        A dictionary of unique routes on country level and their frequencies
        """
    country_sequence = {}
    for df in route_RU_int_NL_matched_imoNr:
    
        ctry_list = df['Country'].tolist()
        ctry_list.append(df.iloc[-1]['Arr_Country'])
        ctry_list = tuple(ctry_list)
        if ctry_list not in list(country_sequence.keys()):
            country_sequence[ctry_list] =  1
        else:
            country_sequence[ctry_list] = country_sequence.get(ctry_list) + 1
    return country_sequence

def depth (givenList):
    """Get number of layers of a multiple nested list
    
    Parameters
    ----------
    givenList : list
        multiple layers of lists in lists.

    Returns
    -------
    int
        Count layers of lists in list.

    """
    for i in givenList:
        if not isinstance(i,list):
            return 1
        else:
            return depth(i)+1
    

# %%
def route_seq_matched_nrimo_par(route_RU_int_NL, alltankers_adjusted,
                            req_nr_port, req_nr_imo, loop, oiltype = 'all', loop_type = 'country'):
    """Get list of routes extracted in human readable text, get frequency of each unique routes on
    different level.
    The total number of ports, and imo within a route can be adjusted by users. 
    
    Parameters
    ----------
    route_RU_int_NL : list
        Contain a list of lists of extracted routes encoded in row indices. These row
        indices originate from the prepocessed port call data..
    alltankers_adjusted : data frame
        Contain port call data of the whole dataset.
        Contain at least columns:
                IMO, DepPort, ArrPort, Country, Arr_Country.
    req_nr_port : int
        Maximun number of ports within a route set by users.
    req_nr_imo : int
        Maximun number of different IMO composing a route set by users.
    loop : boolean (T or F)
        Determined whether routes will contain cyclic routes or not.
    oiltype : string ('crude oil', 'refined oil', 'all'), optional
        Filter routes matched the selected oil type. The default is 'all'.
    loop_type : string ('country', 'port'), optional
        Discard cyclic routes either on country or port level. The default is 'country'.
        Example: routes that contain a country twice will not be included (on country level)

    Raises
    ------
    ValueError
        Empty results are not valid for further implementation.

    Returns
    -------
    route_RU_int_NL_matched_imoNr : list
        A list of data frame containing full routes with start ports,
        intermediate ports, and end ports.
    trip_freq_dict : dict
        Results in frequency for each route segment with in a route.
    country_sequence : dict
        Results in frequency for each unique route on country level.
    port_sequence : dict
        Results in frequency for each unique route on port level.

    """
    req_nr_port = req_nr_port -1
    filtered_dfs = list(filter(lambda lst: len(lst) == req_nr_port, 
                               route_RU_int_NL))
    route_RU_int_NL_matched_imoNr = []
    for lst_ind in filtered_dfs:

        info_shared_port = alltankers_adjusted.loc[lst_ind]
        if loop and (loop_type == 'country'):
            if info_shared_port['Country'].duplicated().any():
                continue
            else:
                if len(info_shared_port['IMO'].unique()) == req_nr_imo: # fill in TOTAL NR. OF PORT ALLOWED
                    route_RU_int_NL_matched_imoNr.append(info_shared_port)

        elif loop and (loop_type == 'port'):
            if info_shared_port['DepPort'].duplicated().any():
                continue
            else:
                if len(info_shared_port['IMO'].unique()) == req_nr_imo: # fill in TOTAL NR. OF PORT ALLOWED
                    route_RU_int_NL_matched_imoNr.append(info_shared_port)

        else:
            if len(info_shared_port['IMO'].unique()) == req_nr_imo: # fill in TOTAL NR. OF PORT ALLOWED
                route_RU_int_NL_matched_imoNr.append(info_shared_port)
    if len(route_RU_int_NL_matched_imoNr) == 0:
        print('No matched results')
    
    
    ship_type = alltankers_adjusted['ShipType'].unique().tolist()
    ship_type_nocrude = [t for t in ship_type if t not in ['Crude Oil Tanker', 'Ore/Oil Carrier']]
    if oiltype != 'all':
        route_RU_int_NL_filtered_crudeoil = []
        route_RU_int_NL_filtered_refinedoil = []
        for df in route_RU_int_NL_matched_imoNr:
            last_row = df.iloc[-1]
            if (last_row['ShipType'] == 'Crude Oil Tanker') or (last_row['ShipType'] == 'Crude/Oil Product Tanker') :
                route_RU_int_NL_filtered_crudeoil.append(df)
            elif last_row['ShipType'] in ship_type_nocrude:
                route_RU_int_NL_filtered_refinedoil.append(df)
    if oiltype == 'crude oil':
        route_RU_int_NL_matched_imoNr = route_RU_int_NL_filtered_crudeoil
        if len(route_RU_int_NL_filtered_crudeoil) <2:
            warnings.warn(f'Not many routes contain crude oil as the last cargo transported to the NL')
        if len(route_RU_int_NL_filtered_crudeoil) == 0:
            raise ValueError("The total number of routes after selecting crude "
                     "oil as the last cargo transported to NL should "
                     "be greater than 0. Suggest to use 'all' as input "
                     "for oiltype")
    elif oiltype == 'refined oil':
        route_RU_int_NL_matched_imoNr = route_RU_int_NL_filtered_refinedoil
        if len(route_RU_int_NL_filtered_refinedoil) <2:
            warnings.warn(f'Not many routes contain refined oil as the last cargo transported to the NL')
        if len(route_RU_int_NL_filtered_refinedoil) == 0:
            raise ValueError("The total number of routes after selecting refined "
                     "oil as the last cargo transported to NL should "
                     "be greater than 0. Suggest to use 'all' as input "
                     "for oiltype")
    
    country_sequence = freq_of_country_seq(route_RU_int_NL_matched_imoNr)
    
    port_sequence = freq_of_port_seq(route_RU_int_NL_matched_imoNr)
    
    # freq of each trips in each route
    trip_freq_dict =  {}
    routes = route_RU_int_NL_matched_imoNr.copy()
    while routes:
        # extract first df 
        df = routes[0]

        matched_port = []
        matched_port.append(df)
        ind = []
        ind.append(0)
        
        # extract all dfs that have the same routes with df (except the first df)

        for row_nx_lst in range(1,len(routes)):
            df_1 = routes[row_nx_lst]
            if (df['DepPort'].tolist() + [df.iloc[-1]['ArrPort']]) ==  (df_1['DepPort'].tolist() + [df_1.iloc[-1]['ArrPort']]):
                matched_port.append(df_1)

                ind.append(row_nx_lst)
              
        lgth = len(df) 
        matched_port = pd.concat(matched_port, ignore_index=True)
        lgth_wholedf = len(matched_port)
        segment_freq = {}
        # if no matched, save the only df directly to a dict
        if lgth == lgth_wholedf:
            port_list = matched_port['DepPort'].tolist()
            port_list.append(matched_port.iloc[-1]['ArrPort'])
            
            comb_port = list(zip(port_list, pd.Series(port_list).shift(-1)))
            comb_port.pop()
            #segment_freq =[(seg[0], seg[1], 1)for seg in comb_port]
            segment_freq = {i: (seg,1) for i, seg in enumerate(
                comb_port)}
            trip_freq_dict[tuple(port_list)] = segment_freq
            # remove indices that counted
            for i in sorted(ind, reverse=True):
                routes.pop(i)
        
        else:
            # get a list of Depport for the route
            port_list = df['DepPort'].tolist()
            port_list.append(df.iloc[-1]['ArrPort'])
            # for each trip( represented per row) in a route, 
            # count total number of unique imos taken that trip across all matched df
            for row_df in range(len(df)):
                # extract rows in the same order accross all matched routes
                # generate index for selected row in a merge df from all matched routes
                ind_list = list(range(row_df, lgth_wholedf, lgth))
                selected_row_df = matched_port.loc[ind_list]
                # if all selected rows are duplicated in terms of IMO, the trip was carried by only one imo
                if len(selected_row_df['IMO'].drop_duplicates()) == 1:
                    de_arr_port = list(selected_row_df.iloc[0, 
                                                            [selected_row_df.columns.get_loc('DepPort'),
                                                             selected_row_df.columns.get_loc('ArrPort')]])
                    segment_freq[row_df] = tuple([tuple(de_arr_port), 1])
                # else more than one IMO using the same trip
                else:
                    de_arr_port = list(selected_row_df.iloc[0, 
                                        [selected_row_df.columns.get_loc('DepPort'),
                                         selected_row_df.columns.get_loc('ArrPort')]])
                    selected_row_df =  selected_row_df['IMO'].drop_duplicates()
                    segment_freq[row_df] = tuple([tuple(de_arr_port), len(selected_row_df)])
                trip_freq_dict[tuple(port_list)] = segment_freq
            # remove indices that counted
            for i in sorted(ind, reverse=True):
                routes.pop(i)
        
    return route_RU_int_NL_matched_imoNr, trip_freq_dict, country_sequence, port_sequence
def extract_route_RU_to_NL_and_others(track_route_fr_RU_to_NL, alltankers_adjusted,
                                      end_port,
                                      port_of_interest, RU_to_NL_con = False):
    """Seperate routes ended at the selected destination and not ended at the selected destination
    

    Parameters
    ----------
    track_route_fr_RU_to_NL : list
        list of all extracted route that met the temporal condition.
    alltankers_adjusted : data frame
        Contain port call data of the whole dataset.
        Contain at least columns:
                IMO, DepPort, ArrPort, Country, Arr_Country.
    end_port : string
        End destinations requested for the extracted routes.
    port_of_interest : list
        Contain ports that users wish to be appeared in the extracted routes.
    RU_to_NL_con : Boolean (T or F), optional
        This is a special case when users want to extract entire routes from RU to NL.
        This option contains strict condition that the algorithm has to follow
        The default is False.
    Russian Dutch case

    Returns
    -------
    route_RU_int_NL : list
        list of dataframe that contains routes ended at the selected destination.
    route_RU_int_other : list
        list of dataframe that contains routes did not end at the selected destination.

    """
    print('Progress in extracting routes to NL:')
    # extract port NL
    route_RU_int_NL = []
    route_RU_int_other = []
    # extract route from RU-aport-NL
    # check hotpot
    for list_ind in tqdm(track_route_fr_RU_to_NL, total = len(track_route_fr_RU_to_NL)):
        df = alltankers_adjusted.loc[list_ind]
        info_shared_port = df.iloc[-1]
        if (info_shared_port['ArrPort'] in (end_port)):
            if RU_to_NL_con:
                if np.isin(df['DepPort'].unique(), (port_of_interest)).any():
                    
                    route_RU_int_NL.append(list_ind)
            else:
                route_RU_int_NL.append(list_ind)
                
        else:
            route_RU_int_other.append(list_ind)
    return route_RU_int_NL, route_RU_int_other



def find_matched_imo_at_1stshared_port(nbs_edges_RU, nr_of_port, nr_imo, port_of_russia,
                                       eu_ports, alltankers_adjusted,
                                       scnd_in_day, lowerbound_time, upperbound_time, RU_to_NL = False):
    """Find connected trips (route segments at the share ports)
    The extracted route is temporal consistent
    

    Parameters
    ----------
    nbs_edges_RU : list
        list of direct neighbours of the selected start port.
    nr_of_port : int
        Maximun number of ports per route.
    nr_imo : int and string ('1' or 'all')
        Users specify the number of unique tankers allowed in extracted routes
        '1' means routes opterated by a single tanker
        'all' means no restriction in number of tankers 
    port_of_russia : list
        ports of start countries.
    eu_ports : list
        ports belong to EU countries.
    alltankers_adjusted : data frame
        Contain port call data of the whole dataset.
        Contain at least columns:
                IMO, DepPort, ArrPort, Country, Arr_Country.
    scnd_in_day : flow
        convert number of dates into second.
    lowerbound_time : int
        unit in hour
        The minimun time requires for time difference between tanker connections
        at share ports.
    upperbound_time : int
        unit in hour
        The maxinum time requires for time difference between tanker connections
        at share ports..
    RU_to_NL : Boolean (T or F)
        Activate functions that follows conditions for extracting the entire routes
        from Russia to NL.The default is False
        Russian Dutch case
    Returns
    -------
    track_route_fr_RU_to_2ndPort_and_connected_IMO : list
        A list of lists, each sublist contains the row indices corresponding to
        each port call, that refers to route segments composing the entire route

    """
    print('Progress in finding matching IMO at the first shared port:')
    track_route_fr_RU_to_2ndPort_and_connected_IMO = []
    for edge in tqdm(nbs_edges_RU, total = len(nbs_edges_RU)):
        #edge = nbs_edges_RU[1]
    
        # extract route from a RU port to its neighbout
        start_at_RU_port = edge[0]
        if RU_to_NL:
            if ((edge[1] in (eu_ports)) or (edge[1] in port_of_russia)):
                next
            
            else: # forloop check the first stop in EU or RU
        
               
                route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_at_RU_port]) &
                                                              alltankers_adjusted['ArrPort'].isin([edge[1]])]
                    
                arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
                # extract all IMO available at the arrival port of the first trip from RU
                diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                    arr_port)]
                
                
                if len(diff_IMO_at_2ndPort)>=1:       
                        
                    for row in range(len(route_from_RUport_to_its_nb)):
                        
                        # arrive time of IMO travel from RU to its nb
                        arr_time_of_IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row,
                                                              route_from_RUport_to_its_nb.columns.get_loc('ArrDate')]
                        IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row:row+1]
                        test_test_ = pd.DataFrame(columns = IMO_fr_RU_to_2ndPort.columns)
                        # loop through all IMO availabel in a nb
                        for row_dep_port in range(len(diff_IMO_at_2ndPort)):
                            # departure time of an IMO
                            dep_time_of_IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                                              diff_IMO_at_2ndPort.columns.get_loc('DepDate')]
                            # dep_port_per_row = diff_IMO_at_2ndPort.iloc[row_dep_port,
                            #                                   diff_IMO_at_2ndPort.columns.get_loc('DepPort')]
                            IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port:
                                                              row_dep_port+1]
                    
                            # time different between IMO from RU and IMO availabe at its nb
                            time_gap = dep_time_of_IMO_avai_at_2ndPort - arr_time_of_IMO_fr_RU_to_2ndPort
                            time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                            # base on threshold assump to take for oil transshipment. if met 
                            # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                            if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= lowerbound_time) & (abs(time_gap_hr) < upperbound_time)):
                                #print(f'time gap after the condition {time_gap_hr}')
                    
                    
                                
                                match_route = pd.concat([IMO_fr_RU_to_2ndPort, IMO_avai_at_2ndPort])
                                if nr_imo == 1:
                                    match_route_imo = match_route['IMO'].unique()
                                    if len(match_route_imo) == 1:
                                        df_merg = list(match_route.index.values)
                                        if len(df_merg) != 0:
                                            
                                            track_route_fr_RU_to_2ndPort_and_connected_IMO.append(df_merg)
                                    else:
                                        next
                                else:       
                                    df_merg = list(pd.concat([IMO_fr_RU_to_2ndPort, IMO_avai_at_2ndPort]).index.values)
            
                                    if len(df_merg) != 0:
                                        
                                        track_route_fr_RU_to_2ndPort_and_connected_IMO.append(df_merg)
                                
                            else:
                                next
        
        
                else: # for matched IMO at shared ports
                    next
    
        else:

            if nr_of_port <= 1:
                route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_at_RU_port]) &
                                                              alltankers_adjusted['ArrPort'].isin([edge[1]])]

                test_test_ = list(route_from_RUport_to_its_nb.index.values)

                track_route_fr_RU_to_2ndPort_and_connected_IMO.append(test_test_)
                
            else:
                
                route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_at_RU_port]) &
                                                              alltankers_adjusted['ArrPort'].isin([edge[1]])]
                imo_from_RU = route_from_RUport_to_its_nb['IMO'].unique()
                    
                arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
                # extract all IMO available at the arrival port of the first trip from RU
                diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                    arr_port)]
                if nr_imo == 1:
                    diff_IMO_at_2ndPort = diff_IMO_at_2ndPort[diff_IMO_at_2ndPort['IMO'].isin(imo_from_RU)]
                
                
                if len(diff_IMO_at_2ndPort)>=1:       
                        
                    for row in range(len(route_from_RUport_to_its_nb)):
                        
                        # arrive time of IMO travel from RU to its nb
                        arr_time_of_IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row,
                                                              route_from_RUport_to_its_nb.columns.get_loc('ArrDate')]
                        IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row:row+1]
                        test_test_ = pd.DataFrame(columns = IMO_fr_RU_to_2ndPort.columns)
                        # loop through all IMO availabel in a nb
                        for row_dep_port in range(len(diff_IMO_at_2ndPort)):
                            # departure time of an IMO
                            dep_time_of_IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                                              diff_IMO_at_2ndPort.columns.get_loc('DepDate')]
                            # dep_port_per_row = diff_IMO_at_2ndPort.iloc[row_dep_port,
                            #                                   diff_IMO_at_2ndPort.columns.get_loc('DepPort')]
                            IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port:
                                                              row_dep_port+1]
                    
                            # time different between IMO from RU and IMO availabe at its nb
                            time_gap = dep_time_of_IMO_avai_at_2ndPort - arr_time_of_IMO_fr_RU_to_2ndPort
                            time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                            # base on threshold assump to take for oil transshipment. if met 
                            # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                            if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= lowerbound_time) & (abs(time_gap_hr) < upperbound_time)):
                                #print(f'time gap after the condition {time_gap_hr}')

                    
                    
                                match_route = pd.concat([IMO_fr_RU_to_2ndPort, IMO_avai_at_2ndPort])
                                if nr_imo == 1:
                                    match_route_imo = match_route['IMO'].unique()
                                    if len(match_route_imo) == 1:
                                        df_merg = list(match_route.index.values)
                                        if len(df_merg) != 0:
                                            
                                            track_route_fr_RU_to_2ndPort_and_connected_IMO.append(df_merg)
                                    else:
                                        next
                                else:       
                                    df_merg = list(pd.concat([IMO_fr_RU_to_2ndPort, IMO_avai_at_2ndPort]).index.values)
                                
                                    if len(df_merg) != 0:
                                        
                                        track_route_fr_RU_to_2ndPort_and_connected_IMO.append(df_merg)
                                
                            else:
                                next
        
        
                else: # for matched IMO at shared ports
                    next
                    

    
    return track_route_fr_RU_to_2ndPort_and_connected_IMO

def find_matched_imo_at_shared_port_noloop_par(route_RU_to_NL,
                                               upperbound_time,nr_imo,  lowerbound_time,
                                               alltankers_adjusted,
                                               scnd_in_day, loop, loop_type = 'country'):
    """Find matching route segments at shared ports (from the second shared ports onwards)
    

    Parameters
    ----------
    route_RU_to_NL : list
        list of routes that did not end at the selected destination.
    nr_imo : int and string ('1' or 'all')
        Users specify the number of unique tankers allowed in extracted routes
        '1' means routes opterated by a single tanker
        'all' means no restriction in number of tankers 
    alltankers_adjusted : data frame
        Contain port call data of the whole dataset.
        Contain at least columns:
                IMO, DepPort, ArrPort, Country, Arr_Country.
    scnd_in_day : flow
        convert number of dates into second.
    lowerbound_time : int
        unit in hour
        The minimun time requires for time difference between tanker connections
        at share ports.
    upperbound_time : int
        unit in hour
        The maxinum time requires for time difference between tanker connections
        at share ports.
    loop : boolean (T or F)
        Determined whether routes will contain cyclic routes or not.
    loop_type : string ('country', 'port'), optional
        Discard cyclic routes either on country or port level. The default is 'country'.
        Example: routes that contain a country twice will not be included (on country level)

    Raises
    ------
    ValueError
        Empty results are not valid.

    Returns
    -------
    track_route_fr_RU_to_NL : list
        list of routes that met the temporal conditions at shared ports.

    """

    track_route_fr_RU_to_NL = []

    for list_ind in route_RU_to_NL:
        # Extract df based on index 
    
        df = alltankers_adjusted.loc[list_ind]
        imo = df['IMO'].unique()
        if nr_imo == 1:
            if len(imo) != 1:
                    raise ValueError('The combination of IMOs in the route sequence contain more than 2 IMO. Check in func'
                                     'find_matched_imo_at_shared_poert_noloop_par')

        arr_port = df['ArrPort'].iloc[-1]
        arr_cntry = df['Arr_Country'].iloc[-1]
        if loop and (loop_type == 'country'):
            if arr_cntry in list(df['Country']):
                continue
            else:
                
                # extract all IMO available at the arrival port of the first trip from RU
                diff_IMO_at_sharedPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                    [arr_port])]
                if nr_imo == 1:
                    diff_IMO_at_sharedPort = diff_IMO_at_sharedPort[diff_IMO_at_sharedPort['IMO'].isin(imo)]
        
                # extract all depports in a potential route collected from the prior iteration
                
                prior_depcntr = df['Country']
                diff_IMO_at_sharedPort = diff_IMO_at_sharedPort[~
                    diff_IMO_at_sharedPort['Arr_Country'].isin(list(prior_depcntr))]
               
                if len(diff_IMO_at_sharedPort)>=1:       
                        
        
                        
                        # arrive time of IMO travel from RU to its nb
                        arr_time_of_IMO_fr_RU_to_2ndPort = df.iloc[-1,
                                                              df.columns.get_loc('ArrDate')]
        
                        # loop through all IMO availabel in a nb
                        for row_dep_port in range(len(diff_IMO_at_sharedPort)):
                            # departure time of an IMO
                            dep_time_of_IMO_avai_at_sharedPort = diff_IMO_at_sharedPort.iloc[row_dep_port,
                                                              diff_IMO_at_sharedPort.columns.get_loc('DepDate')]

                            IMO_avai_at_2ndPort = diff_IMO_at_sharedPort.iloc[row_dep_port:
                                                              row_dep_port+1]
                    
                            # time different between IMO from RU and IMO availabe at its nb
                            time_gap = dep_time_of_IMO_avai_at_sharedPort - arr_time_of_IMO_fr_RU_to_2ndPort
                            time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                            # base on threshold assump to take for oil transshipment. if met 
                            # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                            if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= lowerbound_time) & (abs(time_gap_hr) < upperbound_time)):
                                #print(f'time gap after the condition {time_gap_hr}')
                    
                                match_route = pd.concat([df, IMO_avai_at_2ndPort])
                                if nr_imo == 1:
                                    match_route_imo = match_route['IMO'].unique()
                                    if len(match_route_imo) == 1:
                                        df_merg = list(match_route.index.values)
                                        if len(df_merg) > len(df):
                                            
                                            track_route_fr_RU_to_NL.append(df_merg)
                                    else:
                                        next
                                else:       
                                    df_merg = list(pd.concat([df, IMO_avai_at_2ndPort]).index.values)
            
                                    if len(df_merg) > len(df):
                                        
                                        track_route_fr_RU_to_NL.append(df_merg)
                                
                            else:
                                next
                
        elif loop and (loop_type == 'port'):
            if arr_port in list(df['DepPort']):
                continue
            else:
                # extract all IMO available at the arrival port of the first trip from RU
                diff_IMO_at_sharedPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                    [arr_port])]
                if nr_imo == 1:
                    diff_IMO_at_sharedPort = diff_IMO_at_sharedPort[diff_IMO_at_sharedPort['IMO'].isin(imo)]
        
                # extract all depports in a potential route collected from the prior iteration

                prior_depPort = df['DepPort']
    
                # remove nbs with arrive ports that are repetative to the dep ports of the existing trips

                diff_IMO_at_sharedPort = diff_IMO_at_sharedPort[~
                    diff_IMO_at_sharedPort['ArrPort'].isin(list(prior_depPort))]
                
                if len(diff_IMO_at_sharedPort)>=1:       
                        
                        # arrive time of IMO travel from RU to its nb
                        arr_time_of_IMO_fr_RU_to_2ndPort = df.iloc[-1,
                                                              df.columns.get_loc('ArrDate')]
        
                        # loop through all IMO availabel in a nb
                        for row_dep_port in range(len(diff_IMO_at_sharedPort)):
                            # departure time of an IMO
                            dep_time_of_IMO_avai_at_sharedPort = diff_IMO_at_sharedPort.iloc[row_dep_port,
                                                              diff_IMO_at_sharedPort.columns.get_loc('DepDate')]

                            IMO_avai_at_2ndPort = diff_IMO_at_sharedPort.iloc[row_dep_port:
                                                              row_dep_port+1]
                    
                            # time different between IMO from RU and IMO availabe at its nb
                            time_gap = dep_time_of_IMO_avai_at_sharedPort - arr_time_of_IMO_fr_RU_to_2ndPort
                            time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                            # base on threshold assump to take for oil transshipment. if met 
                            # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                            if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= lowerbound_time) & (abs(time_gap_hr) < upperbound_time)):
                                #print(f'time gap after the condition {time_gap_hr}')
                    
                                
                                match_route = pd.concat([df, IMO_avai_at_2ndPort])
                                if nr_imo == 1:
                                    match_route_imo = match_route['IMO'].unique()
                                    if len(match_route_imo) == 1:
                                        df_merg = list(match_route.index.values)
                                        if len(df_merg) > len(df):
                                            
                                            track_route_fr_RU_to_NL.append(df_merg)
                                    else:
                                        next
                                else:       
                                    df_merg = list(pd.concat([df, IMO_avai_at_2ndPort]).index.values)
            
                                    if len(df_merg) > len(df):
                                        
                                        track_route_fr_RU_to_NL.append(df_merg)
                                
                            else:
                                next
        else:
            # extract all IMO available at the arrival port of the first trip from RU
            diff_IMO_at_sharedPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                [arr_port])]
            if nr_imo == 1:
                diff_IMO_at_sharedPort = diff_IMO_at_sharedPort[diff_IMO_at_sharedPort['IMO'].isin(imo)]
            
            if len(diff_IMO_at_sharedPort)>=1:       
                    
                    # arrive time of IMO travel from RU to its nb
                    arr_time_of_IMO_fr_RU_to_2ndPort = df.iloc[-1,
                                                          df.columns.get_loc('ArrDate')]
    
                    # loop through all IMO availabel in a nb
                    for row_dep_port in range(len(diff_IMO_at_sharedPort)):
                        # departure time of an IMO
                        dep_time_of_IMO_avai_at_sharedPort = diff_IMO_at_sharedPort.iloc[row_dep_port,
                                                          diff_IMO_at_sharedPort.columns.get_loc('DepDate')]

                        IMO_avai_at_2ndPort = diff_IMO_at_sharedPort.iloc[row_dep_port:
                                                          row_dep_port+1]
                
                        # time different between IMO from RU and IMO availabe at its nb
                        time_gap = dep_time_of_IMO_avai_at_sharedPort - arr_time_of_IMO_fr_RU_to_2ndPort
                        time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                        # base on threshold assump to take for oil transshipment. if met 
                        # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                        if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= lowerbound_time) & (abs(time_gap_hr) < upperbound_time)):
                            #print(f'time gap after the condition {time_gap_hr}')
                
                                match_route = pd.concat([df, IMO_avai_at_2ndPort])
                                if nr_imo == 1:
                                    match_route_imo = match_route['IMO'].unique()
                                    if len(match_route_imo) == 1:
                                        df_merg = list(match_route.index.values)
                                        if len(df_merg) > len(df):
                                            
                                            track_route_fr_RU_to_NL.append(df_merg)
                                    else:
                                        next
                                else:       
                                    df_merg = list(pd.concat([df, IMO_avai_at_2ndPort]).index.values)
            
                                    if len(df_merg) > len(df):
                                        
                                        track_route_fr_RU_to_NL.append(df_merg)
                            
                        else:
                            next

    return track_route_fr_RU_to_NL



def filter2(route_RU_int_NL_filtered_v1, alltankers_adjusted, eu_ports
            ):
    """
    Filter the general results
    Function applies certain conditions when users want to extract routes starting 
    from Russia to NL.
    Function assess the hypothesis: if there is any EU port in the route,
     the trips connected to these EU ports have to have the same IMO
    Russian Dutch case

    Parameters
    ----------
    route_RU_int_NL_filtered_v1 : TYPE
        DESCRIPTION.
    alltankers_adjusted : TYPE
        DESCRIPTION.
    eu_ports : TYPE
        DESCRIPTION.

    Returns
    -------
    route_RU_int_NL_filtered_v2 : list
        Routes met the specified conditions:
                when a tanker stopping at an EU country before arriving in NL,
                it will continue operate to NL

    """

    
    route_RU_int_NL_filtered_v2 = []

    for list_ind in route_RU_int_NL_filtered_v1:
        df = alltankers_adjusted.loc[list_ind]
        # check if any depport in the sequence is an EU port
        if df['DepPort'].isin(eu_ports).any():
            depPort = np.array(df['DepPort'])
            # determine the location/index order of the EU port(s) in the route sequence
            mask_euport = np.isin(depPort, eu_ports)
            ind = np.where(mask_euport)[0].tolist()
            if len(ind) == 1: # if only one EU port included
            # check whether the IMO before it the same, if yes, save
                if df['IMO'].iloc[ind[0]] == df['IMO'].iloc[(ind[0]-1)]:
                    route_RU_int_NL_filtered_v2.append(list_ind)
                else: # if not remove
                    next
            else: # if there are more than one EU port in the route sequence
                boo_check = []
                for indx in ind: # check whether the IMO constraint meet for each EU port
                    if df['IMO'].iloc[indx] == df['IMO'].iloc[(indx -1)]:
                        if indx < len(df)-1: # if that EU port not the port connect
                        # or go direct to NL, also check the IMO of the trip after it
                            if df['IMO'].iloc[indx] == df['IMO'].iloc[(indx + 1)]:
                                boo_check.append(True)
                            else:
                                boo_check.append(False)
                                
                        else: 
                            boo_check.append(True)
                    else:
                        boo_check.append(False)
                # if the IMO of all EU ports meet the condition, save the route
                if len(boo_check) > 0 and all(boo_check):
                    route_RU_int_NL_filtered_v2.append(list_ind)
                        
        else: # if not, save the route directly
            route_RU_int_NL_filtered_v2.append(list_ind)
            
    return route_RU_int_NL_filtered_v2
def extract_ports_based_countries(alltankers_adjusted, country):
    """Get all related ports of a country of interest
    

    Parameters
    ----------
    alltankers_adjusted : data frame
        Contain port call data of the whole dataset.
        Contain at least columns:
                IMO, DepPort, ArrPort, Country, Arr_Country.
    country : list
        list of countries of interest.

    Returns
    -------
    port_of_country : list
        list of ports for each selected country.

    """
    port_of_country = []
    for nr in range(len(alltankers_adjusted)):
        if alltankers_adjusted.iloc[nr, 6] in country:
            port = alltankers_adjusted.iloc[nr, 1]
            port_of_country.append(port)
        else:
            next
    port_of_country = list(set(port_of_country))
    return port_of_country

def rename_countries(counter):
    """
    Rename country names

    Parameters
    ----------
    counter : counter
        Dictionary counter containing country names as keys.

    Returns
    -------
    renamed : counter
        Dictionaty counter with new key names.

    """
    renamed = collections.Counter()
    for country, count in counter.items():
        if country == "United States of America":
            country = "USA"
        elif country == "United Kingdom":
            country = "UK"
        renamed[country] += count
    return renamed