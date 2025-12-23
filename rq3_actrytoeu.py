# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 08:00:48 2025

@author: Duyen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:58:57 2025

@author: Duyen
"""
import os
import sys
import time
import itertools
import pandas as pd
from Code import data_processing as pr
import psutil
import joblib
import multiprocess
from  multiprocess import Pool
import tqdm
from memory_profiler import profile
from datetime import datetime
import psutil, os, time, threading
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#@profile
def route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                  win_time_slide, iterat_time, 
                  strike, tot_nr_port, nr_imo, outputpath, Graph_whole_dataset, 
                  port_of_russia, eu_ports, port_of_interest,
                  alltankers_adjusted, ru_country,RU_to_NL, RU_to_NL_con, IMO_con, loop, loop_type):
    """
Function finding all possible routes from givens start ports and end ports. 
The result of extracted routes can vary based on the prior determined rules such as:

+ No EU or RU ports in the first intermediate ports with a start port from a RU port

+ If a route includes an EU port before arriving to an NL port, the same IMO is used to 
transport oil from e.g. a non EU port->EU port->NL port

Extra rules can be set based on the preference of users such as:
+ Connection time between two trips
+ Circular routes 
+ Ship type
+ Ports of interest

Args:
    start_RU_port (list of strings): a list of RU port names where a route begins.
    These ports were selected based on the highest betweeness centrality
    
    end_port (list of strings): a list of NL/or any port names where a route ends.
    These ports were selected based in the highest betweeness centrality
    
    lowerbound_time (int): in hours, the lower bound of the time interval used
    to find matching IMOs at a shared port
    
    upperbound_time (int) : in hours, the upper bound of the time interval used
    to find matching IMO at a shared port
    
    win_time_slide (int): used to extend the interval time (including lower- and
                        and upper-bound). This parameter allows to extend a constant
                        time distance based on the initial defined interval.
                        If this parameter is deactivated, it functions as a iteration
                        counter for a while-loop
                        
    strike (int): in hours. Similar to the win_time_slide, but it allows to extend
    an inconsistent time distance
    
    iterat_time (int): the total number of iteration time will allow to update 
    and exent the lower- and upper-bound of the time interval used to find new 
    matching IMOs at a shared port
    
    tot_nr_port (int): Total number of ports included in a compete route. The
    minimun total number of ports is 3
    
    outputpath (string): path and name of output file
    
    Graph_whole_dataset: a multi-directed graph containing all ports as nodes and 
    all connections between the ports as edges of the whole data. It was constructed
    using networkx package
    
    port_of_russia (list of string): a list of Russian ports
    
    eu_ports (list of string): a list of ports in EU
    
    port_of_interest (list of string): a list of ports belong to the countries of 
    interest. In this case well-known refinery countries. These ports were selected
    based on the highest betweeness centrality
    
    ru_country (string): name of the country
    
    alltankers_adjusted (dataframe): A dataframe contains all ports of call of 
    all IMOs. It requires to have the follow columns:
        Index
        IMO
        DepPort
        ArrPort
        DepDate
        ArrDate
        Country
        Arr_Country
        ShipType
        
    loop (bool): a boolean input with True: remove all circular routes in a route
    with False keeping all types of routes
    
    loop_type (string): there are two types of loops: port and country.
    A route can contain a port/or country more than once in its sequence
    


Returns:
    final_route_RU_to_NL (list of list): a list of index lists. Each index list
    contains a sequence of a full route. To view all data of the routes, using 
    the original data table (alltanker_adjusted)
"""
    # check the value of iterat_time. The maximun number of the iteration depends
    # on the total number of ports requested and the total study period
    # consider time interval is 1 month
    # total_months_in_data = alltankers_adjusted['ArrDate'].sort_values().iloc[-1] - alltankers_adjusted['DepDate'].sort_values().iloc[0]
    # total_months_in_data = round(total_months_in_data.days/30.25, 0)
    # expected_max_iter_nr = total_months_in_data - tot_nr_port
    # if iterat_time > expected_max_iter_nr:
    #     raise ValueError('The total iteration number should be smaller or equal'
    #                      f' {expected_max_iter_nr} to receive a compete output')
        
    # the total number of port has to be >= 3
    
    # if tot_nr_port < 3:
    #     raise ValueError('The total number of ports needs to be larger than '
    #                      'or equal 3')
    # Assess scalable of the algorithm
    runtime_and_mem = pd.DataFrame(columns = ['Nr of Iter', 'Used Mem', 'Run Time'])
    runtime_and_mem_for_smaller_it = pd.DataFrame(columns = ['Nr of Iter', 'Used Mem', 'Used Mem2','Run Time'])
    # a list to collect final result
    routes_comb_w_multi_win_sld = []
    start_iter = 0
    tot_nr_port = tot_nr_port -1
    ori_upperbound_time = upperbound_time
    ori_lowerbound_time = lowerbound_time
    n = start_iter
    m = tot_nr_port
    # define number of core for parallel computation
    processes = os.cpu_count() - 2
    scnd_in_day = 1*24*60*60 #(in seconds)
    start_time = time.time()
    #processes = 6

    while win_time_slide <= iterat_time:
        ts =datetime.fromtimestamp(time.time())
        print(f'Window slide iteration number: {win_time_slide},'
              f' at time{ts.strftime("%Y-%m-%d %H:%M:%S")}')

        if strike == 'None':     
            # update time interval after each iteration
            
            upperbound_time = upperbound_time*win_time_slide
            lowerbound_time = lowerbound_time
            print('upperbound_time', upperbound_time)
            print('lowerbound_time', lowerbound_time)
        else:
            upperbound_time = upperbound_time
            lowerbound_time = lowerbound_time
            print('upperbound_time', upperbound_time)
            print('lowerbound_time', lowerbound_time)
        # collect all edges or neighbours of a start port or multiple start ports
        nbs_edges_RU_cmb = []
        nbs_edges_RU = []
        filtered_final_route_RU_to_NL = []
        if len(start_RU_port) == 1:
            nbs_edges_RU = list(set(list(Graph_whole_dataset.out_edges(start_RU_port,))))
            
        else:
            for ruport in start_RU_port:
                nbs_edges = list(set(list(Graph_whole_dataset.out_edges(ruport,))))
                nbs_edges_RU_cmb.append(nbs_edges)
        
        # unlist nbs edges
        if len(nbs_edges_RU_cmb) > 1:
            for lst in nbs_edges_RU_cmb:
                for sublst in lst:
                    nbs_edges_RU.append(sublst)

       
    # %% Parallel computation code for finding route at the first intermediate port.
    # at the moment it takes longer to compute the result compared to the use of
    # a single core
        # if len(nbs_edges_RU) > processes:
        #     processes = processes
        # else:
        #     processes = 1
        # chunk_size = len(nbs_edges_RU)//processes
        # chunks = [nbs_edges_RU[i:i + chunk_size] for i in range(0, len(nbs_edges_RU), chunk_size)]
        
        
        #     # prepare argument tuples
        # args = [(chunk, port_of_russia,
        # eu_ports, alltankers_adjusted, 
        # scnd_in_day, lowerbound_time, upperbound_time) for chunk in chunks]
        
        # with Pool(processes=processes) as pool:
        #     track_route_fr_RU_to_2ndPort_and_connected_IMO = pool.starmap(
        #         pr.find_matched_imo_at_1stshared_port,
        #         args
        #     )
        # track_route_fr_RU_to_2ndPort_and_connected_IMO = list(
        #     itertools.chain.from_iterable(
        #         track_route_fr_RU_to_2ndPort_and_connected_IMO))
    # %%
    
    # code without parallel computation   
    # extract all possible trips at the first itermediate port that meet the time
    # and ports constraints

        track_route_fr_RU_to_2ndPort_and_connected_IMO = pr.find_matched_imo_at_1stshared_port(
            nbs_edges_RU, tot_nr_port, nr_imo, port_of_russia,eu_ports, 
            alltankers_adjusted, scnd_in_day, lowerbound_time,
            upperbound_time, RU_to_NL)
    
        
        # extract route from RU-hotpot-NL and number of IMO difference            
        route_RU_1int_NL = [] # routes to NL
        route_RU_1int_other = [] # routes not to NL
        # extract route from RU-aport-NL
        for df in track_route_fr_RU_to_2ndPort_and_connected_IMO:

            info_shared_port = alltankers_adjusted.loc[df]
            # the arrival port is in NL?
            if info_shared_port['ArrPort'].iloc[-1] in end_port:
                if RU_to_NL_con:
                    # the route contain port of interest?
                    if (info_shared_port['DepPort'].isin(port_of_interest)).any():
                        
                        route_RU_1int_NL.append(df)
                else:
                    route_RU_1int_NL.append(df)                    
            else: 
                route_RU_1int_other.append(df)
        #* save final routes from RU to NL
        filtered_final_route_RU_to_NL.append(route_RU_1int_NL)   


    
        # delete not necessary variables 
        del  track_route_fr_RU_to_2ndPort_and_connected_IMO
        # iteration calculation, 
        n = n+1
        # get nbs of the next port
        route_RU_to_NL = route_RU_1int_other
    
        if len(route_RU_to_NL) == 0:
            raise ValueError('The total number of possible routes should be'
                             ' larger than 0. It is possible that the'
                             ' time interval is too small, no many available'
                             ' IMO meets the requirements. check route_RU_1int_other')
    
        
        # since Spyder has problems running multiprocessing, this sudo code
        # uses to calibrate some certain parameters of this package to allow it 
        # function normally e.g. adding __spec__ = None
        def f(x):
            return x*x
        if __name__ == '__main__':
            __spec__ = None
            with Pool(5) as p:
                print (p.map(f, [1, 2, 3]))         
               
        
    # phase 2
        while n < m:
            # determine whether using multi-processing or not
            if len(route_RU_to_NL) >processes:
                processes = processes
            else:
                processes = 1

            chunk_size = len(route_RU_to_NL)//processes
            chunks = [route_RU_to_NL[i:i + chunk_size] for i in range(0, len(route_RU_to_NL), chunk_size)]
            
            nr_imo = nr_imo
                # prepare argument tuples
            args = [(chunk, ori_upperbound_time, nr_imo, ori_lowerbound_time, alltankers_adjusted,
                                                            scnd_in_day, loop, loop_type) for chunk in chunks]
            print('Progress in finding matching IMO at a shared port')
            # finding matched trip at the next intermediate ports. Loops are handled
            # in this function

            with Pool(processes=processes) as pool:
                track_route_fr_RU_to_NL = list(tqdm.tqdm(pool.starmap(pr.find_matched_imo_at_shared_port_noloop_par, args), total=len(args)))
            # track_route_fr_RU_to_NL = pr.find_matched_imo_at_shared_port_noloop_par(route_RU_to_NL,
            #                                                 upperbound_time, nr_imo, lowerbound_time,
            #                                                 alltankers_adjusted,
            #                                                 scnd_in_day, loop, loop_type = 'country')
            # ADD Here
            runtime_and_mem.loc[win_time_slide-1, 'Used Mem'] = psutil.virtual_memory().percent

            
            # del args
            if len(track_route_fr_RU_to_NL) == 0:
                raise ValueError('The total number of possible routes after filtering'
                                 ' based on the pre-defined conditions should be larger than 0.'
                                 ' check track_route_fr_RU_to_NL')
            # remove empty list due to unequal division between cores
            track_route_fr_RU_to_NL = [lst for lst in track_route_fr_RU_to_NL if len(lst)>0]
            # unlist
            track_route_fr_RU_to_NL = list(itertools.chain.from_iterable(track_route_fr_RU_to_NL))
            # iter_outputpath = f'./processing/pr_inter_output/iter{win_time_slide-1}.joblib'
            # joblib.dump(track_route_fr_RU_to_NL, iter_outputpath)
            # extract routes to NL
            route_RU_int_NL, route_RU_int_other = pr.extract_route_RU_to_NL_and_others(
                track_route_fr_RU_to_NL, alltankers_adjusted,
                end_port,
                port_of_interest, RU_to_NL_con)
            if len(route_RU_int_NL) == 0:
                raise ValueError('The total number of possible routes from RU to NL'
                                 ' has to be greater than 0. It is possible that the'
                                 ' time interval is too small, not many available'
                                 ' IMOs meet the requirements.'
                                 ' Check route_RU_int_NL ')
         
            # if loop allowed and do not allow a direct trip from RU to NL within 
            # the routes sequence, filter those routes out

            # if len(route_RU_int_NL) >processes:
            #     processes = processes
            # else:
            #     processes = 1
                 
            # chunk_size = len(route_RU_int_NL)//processes
            # #print('chunk_size of iter', n, chunk_size, 'en length of the whole route', len(route_RU_int_NL))
            # chunks = [route_RU_int_NL[i:i + chunk_size] for i in range(0, len(route_RU_int_NL), chunk_size)]
            # # prepare argument tuples
            # args = [(chunk, alltankers_adjusted, ru_country, port_of_russia) for chunk in chunks]
            
            # print('Progress in filter 1')
            # with Pool(processes=processes) as pool:
            #     route_RU_int_NL_filtered_v1 = list(tqdm.tqdm(pool.starmap(pr.filter1, args), total=len(args)))
                
            # if len(route_RU_int_NL_filtered_v1) == 0:
            #     raise ValueError('The total number of possible routes from RU to NL'
            #                      ' has to be greater than 0. It is possible that'
            #                      ' the time interval is too small, not many available'
            #                      ' IMOs meets the requirements. No routes match the'
            #                      ' requirements, after filtering all'
            #                      ' routes containing a RU port in the sequence going'
            #                      ' direct to NL')
              
            # # remove empty list
            # route_RU_int_NL_filtered_v1 = [lst for lst in route_RU_int_NL_filtered_v1 if len(lst)>0]
            # # unlist
            # route_RU_int_NL_filtered_v1 = list(itertools.chain.from_iterable(route_RU_int_NL_filtered_v1))
            # del chunk_size, chunks, args
            # identify number of cores
            if IMO_con:
                if len(route_RU_int_NL) >processes:
                    processes = processes
                else:
                    processes = 1

                # Remove routes do not match the IMO constrainst: if there is any EU port in the route,
                 # the trips connected to these EU ports have to have the same IMO
                chunk_size = len(route_RU_int_NL)//processes
                chunks = [route_RU_int_NL[i:i + chunk_size] for i in range(0, len(route_RU_int_NL), chunk_size)]
                args = [(chunk,  alltankers_adjusted, eu_ports) for chunk in chunks]
                
                print('Progress in filter 2')
                with Pool(processes=processes) as pool:
                    route_RU_int_NL_filtered_v2 = list(tqdm.tqdm(pool.starmap(pr.filter2, args), total=len(args)))
                del chunk_size, chunks, args
                
                if len(route_RU_int_NL_filtered_v2) == 0:
                    raise ValueError(' The total number of possible routes from RU to NL'
                                     ' has to be greater than 0. It is possible that'
                                     ' the defined time interval is too small, not many available'
                                     ' IMOs meets the requirements. No routes match the'
                                     ' requirements, after filtering all'
                                     ' routes that contain different IMOs from an EU country'
                                     ' to NL')
            
                
                route_RU_int_NL_filtered_v2 = [lst for lst in route_RU_int_NL_filtered_v2 if len(lst)>0]
                route_RU_int_NL_filtered_v2 = list(itertools.chain.from_iterable(route_RU_int_NL_filtered_v2))
                
                filtered_final_route_RU_to_NL.append(route_RU_int_NL_filtered_v2)
                if len(route_RU_int_NL_filtered_v2) == 0:
                    # displaying the warning
                    warnings.warn(f'The total number of possible routes from RU to NL'
                                  f' after all filter equal 0 when total number of port'
                                  f' reach {tot_nr_port + 2}')
                # update list of routes for the next round iteration
                route_RU_to_NL = route_RU_int_other
                del route_RU_int_NL_filtered_v2, route_RU_int_NL
        
                if len(route_RU_int_other) == 0:
                    raise ValueError('The total number of possible routes for the'
                                     ' next iteration should be larger than 0')
            else:
                filtered_final_route_RU_to_NL.append(route_RU_int_NL)
    
            #del chunk_size, chunks

            # update iteration
            n = n+1

        run_time = (time.time() - start_time)
        runtime_and_mem.loc[win_time_slide-1, 'Nr of Iter'] = win_time_slide
        
        runtime_and_mem.loc[win_time_slide-1, 'Run Time'] = run_time
        # can DELETE late for this save
        namecsv = f'./processing/pr_inter_output/{len(start_RU_port)}_time4w_nrtotport_{tot_nr_port+1}.csv'
        # namecsv = 'performance_allRUport__timeinf_nrtotport5.csv'
        runtime_and_mem.to_csv(namecsv)

        # update time interval after each window slide iteration
        if strike == 'None':
            lowerbound_time = upperbound_time
            upperbound_time = ori_upperbound_time
            win_time_slide = win_time_slide+1
        else:
           lowerbound_time = upperbound_time
           upperbound_time = upperbound_time + strike
           win_time_slide = win_time_slide+1
          
        routes_comb_w_multi_win_sld.append(filtered_final_route_RU_to_NL)
        # can DELETE late for this save
        joblib.dump(routes_comb_w_multi_win_sld, outputpath)

        # restart number of iteration after each window slide iteration
        n = start_iter
        m = tot_nr_port

    if len(routes_comb_w_multi_win_sld) == 0:
        raise ValueError(' No routes were found')
    # combine all results from different time window slide
    if len(routes_comb_w_multi_win_sld) >1:
        for lst in range(len(routes_comb_w_multi_win_sld)):
            final_route_RU_to_NL = list(itertools.chain.from_iterable(routes_comb_w_multi_win_sld))
            list_depth = pr.depth(final_route_RU_to_NL)-2
            for lst in range(list_depth):
                final_route_RU_to_NL = list(itertools.chain.from_iterable(final_route_RU_to_NL))
            
    else:
        list_depth = pr.depth(routes_comb_w_multi_win_sld)-2
        final_route_RU_to_NL = routes_comb_w_multi_win_sld
        for lst in range(list_depth):
            final_route_RU_to_NL = list(itertools.chain.from_iterable(final_route_RU_to_NL))

    # save the final output
    joblib.dump(final_route_RU_to_NL, outputpath)
    namecsv = f'./processing/pr_inter_output/{len(start_RU_port)}_time4w_nrtotport_{tot_nr_port+1}.csv'
    # namecsv = 'performance_allRUport__timeinf_nrtotport5.csv'
    runtime_and_mem.to_csv(namecsv)
    return final_route_RU_to_NL
