# from VI import fast_value_iteration
import ray
from cunard_competition import fast_value_iteration
from frozen_lake import *
import sys
import argparse
import time


parser = argparse.ArgumentParser(description='fastVI')
parser.add_argument('-m','--map_size', type=int, default=64, help='Size of generated map')
parser.add_argument('-t','--type',  type=str, default="sync", help='sync or async ?')
parser.add_argument('-w','--workers', type= int , default = 4, help = 'numbers of workers')
args = parser.parse_args()

#Debug Code
print(args.map_size,args.type,args.workers)
# input()

sync_flag = True if args.type == "sync" else False

#Define map and environment
MAPMAP = generate_map((args.map_size,args.map_size))
frozen_env = FrozenLakeEnv(desc =MAPMAP )
input("Map generation complete")

st = time.time()
v, pi = fast_value_iteration(env = frozen_env, workers_num = args.workers)
et = time.time()

save_and_print_results(v,pi, map_size=args.map_size,env= frozen_env, beta = 0.99,name="fastVI")
print("time taken by asynchronous:",et-st)





# # Run Value Iteration
# st = time.time()
# v, pi = fast_value_iteration(env = frozen_env, workers_num = args.workers,synchronous = False,)
# et = time.time()

# save_and_print_results(v,pi, map_size=args.map_size,env= frozen_env, beta = 0.99,name="fastVI")

# async_time = et-st
# print("time taken by asynchronous:",et-st)
# input()

# #Run Value Iteration
# st = time.time()
# v, pi = fast_value_iteration(env = frozen_env, workers_num = args.workers,synchronous = sync_flag,)
# et = time.time()

# save_and_print_results(v,pi, map_size=args.map_size,env= frozen_env, beta = 0.99,name="fastVI")

# sync_time = et-st
# print("time taken",et-st)
# # input()



# print("Async  - sync :",async_time -sync_time)