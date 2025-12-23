# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:26:25 2025

@author: Duyen
"""
import pandas as pd
import plotly.express as px
pd.DataFrame.iteritems = pd.DataFrame.items
import plotly.io as pio
import matplotlib.pyplot as plt
import collections
import numpy as np
import seaborn as sns
import squarify
import os


def treemap(data, name):
    # Count occurrences
    if isinstance(data, dict):
        count_nat = collections.Counter(data.values())
    else:
        count_nat = collections.Counter(data)
    
    # Create dataframe
    df = pd.DataFrame({
        'labels': list(count_nat.keys()),
        'values': list(count_nat.values())
    })
    
    # Sort from big to small
    df = df.sort_values(by="values", ascending=False).reset_index(drop=True)
    
    # Color palette
    colors = [
        '#fae588', '#f79d65', '#f9dc5c', '#e8ac65', '#e76f51', '#ef233c', '#b7094c',
        '#7f4f24', '#a44a3f', '#d62828', '#f77f00', '#fcbf49', '#eae2b7', '#6a994e',
        '#386641', '#2a9d8f', '#264653', '#457b9d', '#1d3557', '#a8dadc', '#e63946',
        '#ffb4a2', '#e5989b', '#b5838d', '#6d6875'
    ]
    
    # Seaborn style
    sns.set_style(style="whitegrid")
    
    # Prepare labels with values
    labels_with_values = [f"{l}\n{v}" for l, v in zip(df["labels"], df["values"])]
    
    # Plot
    plt.figure(figsize=(10, 6))
    squarify.plot(
        sizes=df["values"].values,
        label=labels_with_values,
        alpha=0.6,
        color=colors[:len(df)],  # match number of colors
        text_kwargs={'fontsize': 35, 'weight': 'bold'}
    )
    plt.title(name, fontsize=25, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def par_plot(data, title, color_based, dim= True):

    
    
    # Parallel visualization
    
    parplot_NL_to_RU_direct = data[[
        'DepPort', 'Country', 'Arr_Country', 'ArrPort']].rename(columns={
        'DepPort': 'Dep. Ports',
        'Country': 'Dep. Countries',
        'Arr_Country': 'Arr. Countries',
        'ArrPort': 'Arr. Ports'
    })
    
    pio.renderers.default = "browser"
    
    # Convert Arr. Nation to categorical and numeric codes
    parplot_NL_to_RU_direct['Nation_code'] = (
        parplot_NL_to_RU_direct[color_based].astype('category').cat.codes
    )
    # Parallel categories plot
    fig = px.parallel_categories(
        parplot_NL_to_RU_direct,
        dimensions=['Dep. Ports', 'Dep. Countries', 'Arr. Countries', 'Arr. Ports'],
        color='Nation_code',
        color_continuous_scale=px.colors.qualitative.Safe  # qualitative palette
    )
    
    # Adjust layout: text size, figure size, and margins
    fig.update_layout(
        title=title,
        font=dict(size=80),    # text size
        width=2800,            # diagram width
        height=1200,            # diagram height
        margin=dict(l=800, r=400, t=100, b=50)  # left, right, top, bottom
    )
    
    # Hide the color bar / legend
    fig.update_coloraxes(showscale=False)
    if dim:
        for d in fig.data[0].dimensions:
            d.label = ""
    
    fig.show()
    
def sim_par_plot(data, title, color_based):

    
    
    # Parallel visualization
    
    parplot_NL_to_RU_direct = data[[
        'DepPort', 'Country', 'Arr_Country', 'ArrPort']].rename(columns={
        'DepPort': 'Dep. Ports',
        'Country': 'Dep. Countries',
        'Arr_Country': 'Arr. Countries',
        'ArrPort': 'Arr. Ports'
    })
    
    pio.renderers.default = "browser"
    
    # Convert Arr. Nation to categorical and numeric codes
    parplot_NL_to_RU_direct['Nation_code'] = (
        parplot_NL_to_RU_direct[color_based].astype('category').cat.codes
    )
    # Parallel categories plot
    fig = px.parallel_categories(
        parplot_NL_to_RU_direct,
        dimensions=[ 'Dep. Countries', 'Arr. Countries'],
        color='Nation_code',
        color_continuous_scale=px.colors.qualitative.Safe  # qualitative palette
    )
    
    # Adjust layout: text size, figure size, and margins
    fig.update_layout(
        title=title,
        font=dict(size=50),    # text size
        width=1500,            # diagram width
        height=800,            # diagram height
        margin=dict(l=400, r=600, t=100, b=50)  # left, right, top, bottom
    )
    
    # Hide the color bar / legend
    fig.update_coloraxes(showscale=False)
    
    fig.show()


def stackbar_w_gr_plot(data, col1,col2, nameplot, top_nr, ylabel, xlabel, others = True): #col1=Country, col2='nr_port' preveously
    data =  data.groupby([col1, col2]).size().unstack(fill_value=0)
    data['sum'] = data.sum(axis = 1)
    data = data.sort_values(by = 'sum', ascending = False)
    top = data[:top_nr]
    if others:
        others_sum = data[top_nr:].sum(axis = 0)
        other_df = pd.DataFrame([others_sum], index = ['Others'])
    
        top = pd.concat([top, other_df])
    
    
    # Drop the 'sum' column for stacking
    top = top.drop(columns="sum")
    # Plot stacked bar
    top.plot(kind="bar", stacked=True, figsize=(10, 10))
    #plt.title(nameplot, fontsize=40, pad=30)
    plt.ylabel(ylabel, fontsize=40)
    plt.xlabel(xlabel, fontsize=45)
    plt.xticks(rotation=45, ha="right", fontsize=45)
    plt.yticks(fontsize=30)
    
    # ✅ Legend inside (example: upper right corner)
    plt.legend(
        fontsize=25,
        loc="upper right",   # choose position inside
        frameon=True         # optional: add box background
    )
    
    plt.tight_layout()
    plt.show()


def stackbar_plot(data, nameplot, ylabel, xlabel):
    # Plot stacked bar
    data.plot(kind="bar", stacked=True, figsize=(10, 10))

    plt.title(nameplot, fontsize = 40, pad = 30)
    plt.ylabel(ylabel, fontsize = 30)
    plt.xlabel(xlabel, fontsize =30)
    plt.xticks(rotation=45, ha="right",fontsize = 30 )
    plt.yticks(fontsize = 30)
    plt.legend(fontsize = 30)

    # Move legend outside
    plt.legend(fontsize=30, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
def bar_top_plot(data, top_nr, nameplot):
    cntry_bf_NL_RU_notdirect = sorted(data.items(), key=lambda x: x[1], reverse=True)
    top_cntry_bf_NL_RU_notdirect = cntry_bf_NL_RU_notdirect[:top_nr]

    others_sum = sum(v for _, v in cntry_bf_NL_RU_notdirect[top_nr:])
    top_cntry_bf_NL_RU_notdirect.append(("Others", others_sum))

    # Separate countries and counts
    countries = [c for c, _ in top_cntry_bf_NL_RU_notdirect]
    counts = [v for _, v in top_cntry_bf_NL_RU_notdirect]
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(countries, counts, color='steelblue')

    # Formatting
    ax.set_ylabel('Counts', fontsize=30)
    ax.set_title(nameplot, fontsize=30, pad=30)
    ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=25)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    plt.tight_layout()
    plt.show()
    
def scalability_test(patterns,mem_patterns, path, name1, name2, second, nopar):    
    # Read all files by pattern into a dictionary of DataFrame dicts
    dfs_dict = {}
    
    for prefix, var_name in patterns.items():
        matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
        dfs_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}
        
    dfs_mem_dict = {}
    for prefix, var_name in mem_patterns.items():
        matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
        dfs_mem_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}
    
    runtime_colect = []
    for key, val  in dfs_dict.items():
        dict_2 = dfs_dict[key]
        runtime_pertime = []
        for key2, val2 in dict_2.items():
            runtime = val2.iloc[-1,3]
            runtime_pertime.append(runtime)
        runtime_colect.append(runtime_pertime)
    
    max_len = [len(lst) for lst in runtime_colect]
    max_len = max(max_len)
    # Pad each list with np.nan to match max_len
    if second:   
        # keep second
        for lst in range(len(runtime_colect)):
            runtime_colect[lst] = [ round(x/1,0) for x in runtime_colect[lst]]
    else: # convert to min
        for lst in range(len(runtime_colect)):
            runtime_colect[lst] = [ round(x/60,0) for x in runtime_colect[lst]]
    # Pad each list with np.nan to match max_len
    runtime_colect = [lst + [np.nan] * (max_len - len(lst)) for lst in runtime_colect]
    max_mem = []   
    for key, val  in dfs_mem_dict.items():
        dict_2 = dfs_mem_dict[key]
        maxmem_pertime = []
        for key2, val2 in dict_2.items():
            mem = val2['Process Memory (MB)'].max()
            maxmem_pertime.append(mem)
        max_mem.append(maxmem_pertime)
            
    max_mem = [lst + [np.nan] * (max_len - len(lst)) for lst in max_mem]
    
    # x-axis starting from 3
    from matplotlib.ticker import MultipleLocator
    

    x = np.arange(3, 3 + max_len)
    if nopar == 'gen':
        plt.figure(figsize=(11,8))
        plt.plot(x, runtime_colect[0], marker='o', label='1w', linewidth=3)
        plt.plot(x, runtime_colect[1], marker='o', label='2w', linewidth=3)
        plt.plot(x, runtime_colect[2], marker='o', label='3w', linewidth=3)
        plt.plot(x, runtime_colect[3], marker='o', label='4w', linewidth=3)
        # plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop', linewidth=3)
    elif nopar == 'input':
        plt.figure(figsize=(11,8))
        plt.plot(x, runtime_colect[0], marker='o', label='1RU', linewidth=3)
        plt.plot(x, runtime_colect[1], marker='o', label='2RU', linewidth=3)
        plt.plot(x, runtime_colect[2], marker='o', label='3RU', linewidth=3)
        plt.plot(x, runtime_colect[3], marker='o', label='4RU', linewidth=3)
    elif nopar == '1imo':
        x = np.arange(2, 2 + max_len)
        plt.figure(figsize=(10,7))
        plt.plot(x, runtime_colect[0], marker='o', linewidth=3)
        
    else: 
        plt.figure(figsize=(11,8))
        plt.plot(x, runtime_colect[0], marker='o', label='4 cores', linewidth=3)
        plt.plot(x, runtime_colect[1], marker='o', label='6 cores', linewidth=3)
        plt.plot(x, runtime_colect[2], marker='o', label='8 cores', linewidth=3)
        plt.plot(x, runtime_colect[3], marker='o', label='10 cores', linewidth=3)
        plt.plot(x, runtime_colect[4], marker='o', label='12 cores', linewidth=3)
    
    plt.xlabel("Total number of ports", fontsize=30)
    if second:
        plt.ylabel("Runtime (second)", fontsize=30)
    else:
        plt.ylabel("Runtime (minutes)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.title("Runtime Comparison", fontsize=30)
    plt.legend(fontsize=24)
    plt.grid(True)
    
    # Increase tick font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set x-axis tick interval to 1
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    plt.show()
    plt.savefig(name1, format='pdf')
    
    x = np.arange(3, 3 + max_len)
    if nopar == 'gen':
        plt.figure(figsize=(11,8))
        plt.plot(x, max_mem[0], marker='o', label='1w', linewidth=3)
        plt.plot(x, max_mem[1], marker='o', label='2w',linewidth=3)
        plt.plot(x, max_mem[2], marker='o', label='3w',linewidth=3)
        plt.plot(x, max_mem[3], marker='o', label='4w',linewidth=3)
        #plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
    elif nopar == 'input':
        
        plt.figure(figsize=(11,8))
        plt.plot(x, max_mem[0], marker='o', label='1RU', linewidth=3)
        plt.plot(x, max_mem[1], marker='o', label='2RU', linewidth=3)
        plt.plot(x, max_mem[2], marker='o', label='3RU', linewidth=3)
        plt.plot(x, max_mem[3], marker='o', label='4RU', linewidth=3)
    elif nopar == '1imo':
        x = np.arange(2, 2 + max_len)
        plt.figure(figsize=(10,7))
        plt.plot(x, max_mem[0], marker='o', linewidth=3)
    else:
        plt.figure(figsize=(11,8))

        plt.plot(x, max_mem[0], marker='o', label='4 cores', linewidth=3)
        plt.plot(x, max_mem[1], marker='o', label='6 cores', linewidth=3)
        plt.plot(x, max_mem[2], marker='o', label='8 cores', linewidth=3)
        plt.plot(x, max_mem[3], marker='o', label='10 cores', linewidth=3)
        plt.plot(x, max_mem[4], marker='o', label='12 cores', linewidth=3)
    
    plt.xlabel("Total number of ports", fontsize=30)
    plt.ylabel("Used memory (MiB)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.title("Used memory Comparison", fontsize=30)
    plt.legend(fontsize=24)
    plt.grid(True)
    
    # Increase tick font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set x-axis tick interval to 1
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    plt.show()
    plt.savefig(name2, format='pdf')

