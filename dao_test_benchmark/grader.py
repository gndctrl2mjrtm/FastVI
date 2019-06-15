from __future__ import division
import ray
import time
import pickle
from pickle import load
import matplotlib.pyplot as plt
from copy import deepcopy
from random import randint, choice
from subprocess import call
import frozenlake
import VI as competition
import numpy as np
import signal
from contextlib import contextmanager
import os
import sys

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# pkl_prefix = ''
# if len(sys.argv) > 1:
#     pkl_prefix = sys.argv[1] + '/'

# print('--looking up', pkl_prefix + 'dist_vi_v2_8_pi.pkl')
# try:
#     x = load(open(pkl_prefix + 'dist_vi_v2_8_pi.pkl', 'rb'))
#     y = load(open('../pkls/dist_vi_v2_8_pi.pkl', 'rb'))
#     print('--policy is correct? ', np.array_equal(x, y))
# except FileNotFoundError:
#     print('--no dist_vi_v2_8 found!!!')

MAP = (frozenlake.MAPS["16x16"], 16)
map_size = MAP[1]
run_time = {}
env = frozenlake.FrozenLakeEnv(desc = MAP[0], is_slippery = True)

ray.init(include_webui=False, ignore_reinit_error=True,
        redis_max_memory= 100 * 1000 * 1000,
        object_store_memory= 5 * 1000 * 1000 * 1000)
time_16 = 0
for i in range(3):
    try:
        with time_limit(10): # our V2 runs in 6.9 for 16x16 map 
            start_time = time.time()
            v, pi = competition.fast_value_iteration(env, beta = 0.99)
            end_time = time.time()
            v_np, pi_np  = np.array(v), np.array(pi)
            pickle.dump(pi_np, open('policy_16.pkl', 'wb'))
            time_16 += end_time - start_time
    except TimeoutException as e:
        print("Time out!!!!!!!!!!!!")
        time_16 += 10



print("time_16:", time_16/3)
print("\n")
