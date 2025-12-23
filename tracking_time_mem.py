#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:13:56 2025

@author: duyen
"""


import threading
import time
import psutil
import os
import csv
from memory_profiler import memory_usage
import joblib

def log_system_memory(interval, stop_event, system_log):
    """Log system memory usage periodically into a shared list."""
    process = psutil.Process(os.getpid())
    start_time = time.time()

    while not stop_event.is_set():
        elapsed = time.time() - start_time
        rss_mb = process.memory_info().rss / (1024 * 1024)
        sys_percent = psutil.virtual_memory().percent
        system_log.append((elapsed, rss_mb, sys_percent))
        time.sleep(interval)

def run_with_dual_memory_tracking(func, func_args=(), log_interval=1.0
                                  ):
    """Run a function while tracking both system and process memory usage."""
    # Shared log for system memory usage
    system_memory_log = []

    # Start system memory logging in background
    stop_event = threading.Event()
    logger_thread = threading.Thread(target=log_system_memory,
                                     args=(log_interval, stop_event, system_memory_log))
    logger_thread.start()

    # Track process memory usage using memory_profiler
    proc_memory_log = memory_usage((func, func_args), interval=log_interval, retval=True)

    # Stop system memory logging
    stop_event.set()
    logger_thread.join()

    # Extract result from proc_memory_log (last element if retval=True)
    if isinstance(proc_memory_log, tuple):
        process_mem_data, result = proc_memory_log
    else:
        process_mem_data, result = proc_memory_log, None


    return result, system_memory_log, process_mem_data



