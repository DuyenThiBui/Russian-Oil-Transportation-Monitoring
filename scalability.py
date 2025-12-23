# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:43:35 2025

@author: Duyen
This script generates the results for scalability assessment, focusing on latency and memory usage
We tested the scalability when varied different parameters such as time inverval, number of start ports, numbers of cores, 
restricted to routes operated by a single tanker.
"""

import os
cwd = os.getcwd()
os.chdir(cwd)

from Code import plot as plt_cus



path = r'D:\Dropbox\Duyen\University\Master\Year 2\Internship\processing\pr_inter_output'

# Prefix patterns mapped to variable names
# timeinterval
patterns = {
    "Performance_loop_nrRU_1_time1w": "dfs_loop_1w1RU",
    "Performance_loop_nrRU_1_time2w": "dfs_loop_2w1RU",
    "Performance_loop_nrRU_1_time3w": "dfs_loop_3w1RU",
    "Performance_loop_nrRU_1_time4w": "dfs_loop_4w1RU"
}
patterns_noloop = {
    "Performance_noloop_nrRU_1_time1w": "dfs_loop_1w1RU",
    "Performance_noloop_nrRU_1_time2w": "dfs_loop_2w1RU",
    "Performance_noloop_nrRU_1_time3w": "dfs_loop_3w1RU",
    "Performance_noloop_nrRU_1_time4w": "dfs_loop_4w1RU"
}
# number of start ports
patterns_dff_input = {
    "Performance_noloop_nrRU_1_time4w": "dfs_loop_1w1RU",
    "Performance_noloop_nrRU_2_time4w": "dfs_loop_2w1RU",
    "Performance_noloop_nrRU_3_time4w": "dfs_loop_3w1RU",
    "Performance_noloop_nrRU_4_time4w": "dfs_loop_4w1RU"
}
# number of cores
patterns_par_core = {
    "Performance_loop_pr_4_nrRU_1_time4w": "dfs_loop_pr4",
    "Performance_loop_pr_6_nrRU_1_time4w": "dfs_loop_pr6",
    "Performance_loop_pr_8_nrRU_1_time4w": "dfs_loop_pr8",
    "Performance_loop_pr_10_nrRU_1_time4w": "dfs_loop_pr10",
    "Performance_loop_pr_12_nrRU_1_time4w": "dfs_loop_pr12"

    }
patterns_noloop_par_core = {
    "Performance_noloop_pr_4_nrRU_1_time4w": "dfs_loop_pr4",
    "Performance_noloop_pr_6_nrRU_1_time4w": "dfs_loop_pr6",
    "Performance_noloop_pr_8_nrRU_1_time4w": "dfs_loop_pr8",
    "Performance_noloop_pr_10_nrRU_1_time4w": "dfs_loop_pr10",
    "Performance_noloop_pr_12_nrRU_1_time4w": "dfs_loop_pr12"

    }
# resstricted to one imo
patterns_1imo = {
    "Performance_1imo": "dfs_loop_1w1RU",
    
}
mem_patterns = {
    "proc_log_file_loop_RU_1_1w": "dfs_loop_1w1RU",
    "proc_log_file_loop_RU_1_2w": "dfs_loop_2w1RU",
    "proc_log_file_loop_RU_1_3w": "dfs_loop_3w1RU",
    "proc_log_file_loop_RU_1_4w": "dfs_loop_4w1RU"
}
mem_patterns_noloop = {
    "proc_log_file_noloop_RU_1_1w": "dfs_loop_1w1RU",
    "proc_log_file_noloop_RU_1_2w": "dfs_loop_2w1RU",
    "proc_log_file_noloop_RU_1_3w": "dfs_loop_3w1RU",
    "proc_log_file_noloop_RU_1_4w": "dfs_loop_4w1RU"
}
mem_diff_input = {
    "proc_log_file_noloop_RU_1_4w": "dfs_loop_1w1RU",
    "proc_log_file_noloop_RU_2_4w": "dfs_loop_2w1RU",
    "proc_log_file_noloop_RU_3_4w": "dfs_loop_3w1RU",
    "proc_log_file_noloop_RU_4_4w": "dfs_loop_4w1RU"
}
mem_par_core = {
    "proc_log_file_loop_pr_4_RU_1_4w": "dfs_loop_pr4",
    "proc_log_file_loop_pr_6_RU_1_4w": "dfs_loop_pr6",
    "proc_log_file_loop_pr_8_RU_1_4w": "dfs_loop_pr8",
    "proc_log_file_loop_pr_10_RU_1_4w": "dfs_loop_pr10",
    "proc_log_file_loop_pr_12_RU_1_4w": "dfs_loop_pr12"

}

mem_noloop_par_core = {
    "proc_noloop_log_file_loop_pr_4_RU_1_4w": "dfs_loop_pr4",
    "proc_noloop_log_file_loop_pr_6_RU_1_4w": "dfs_loop_pr6",
    "proc_noloop_log_file_loop_pr_8_RU_1_4w": "dfs_loop_pr8",
    "proc_noloop_log_file_loop_pr_10_RU_1_4w": "dfs_loop_pr10",
    "proc_noloop_log_file_loop_pr_12_RU_1_4w": "dfs_loop_pr12"

}
mem_1imo = {
    "proc_log_file_1imo": "dfs_loop_pr4",


}



patterns_noloop_all = {
    "all_Performance_noloop_nrRU_1_time1w": "dfs_loop_1w1RU",
    "all_Performance_noloop_nrRU_1_time2w": "dfs_loop_2w1RU",
    "all_Performance_noloop_nrRU_1_time3w": "dfs_loop_3w1RU",
    "all_Performance_noloop_nrRU_1_time4w": "dfs_loop_4w1RU"
}

mem_patterns_noloop_all = {
    "all_proc_log_file_noloop_RU_1_1w": "dfs_loop_1w1RU",
    "all_proc_log_file_noloop_RU_1_2w": "dfs_loop_2w1RU",
    "all_proc_log_file_noloop_RU_1_3w": "dfs_loop_3w1RU",
    "all_proc_log_file_noloop_RU_1_4w": "dfs_loop_4w1RU"
}
# ploting
plt_cus.scalability_test(patterns_1imo,mem_1imo, path,
                         "./screenshots/scal_1imo_time_s.pdf", "./screenshots/scal_1imo_mem.pdf", second = True, nopar = '1imo')
