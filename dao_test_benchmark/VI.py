import ray
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice
import pickle
import numpy as np

ray.shutdown()
ray.init()

# A : Action Space 
# S : State Space 

@ray.remote
class VI_worker_class(object):
    def __init__(self, A, successors, rewards, beta, start_state, end_state):
        self.A = A
        self.succ_arr = successors
        self.rew_arr = rewards
        self.beta = beta
        self.start_state = start_state
        self.end_state = end_state

    def compute(self, values):
        return_vals = [(float('-inf'), 0) for i in range(self.end_state-self.start_state)]
        max_error = 0
        for state in range(self.start_state, self.end_state):
            for action in range(self.A):
                expected_val = 0
                successors = self.succ_arr[state - self.start_state][action]
                for (next_state, prob) in successors:
                    expected_val += prob * values[next_state]
                new_val = self.rew_arr[state - self.start_state][action] + self.beta*expected_val
                if new_val > return_vals[state - self.start_state][0]:
                    return_vals[state - self.start_state] = (new_val, action)
            error = abs(return_vals[state - self.start_state][0] - values[state])
            if error > max_error:
                max_error = error
        return return_vals, max_error, self.start_state

def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4, synchronous = False):
    # Make VI server
    S = env.GetStateSpace()
    A = env.GetActionSpace()
    v_current = [0] * S
    pi = [0] * S
    batch_indxs = [(int(i*(S/workers_num)), int((i+1)*(S/workers_num))) for i in range(workers_num)]
#     print(batch_indxs) #[(0, 100), (100, 200), (200, 300), (300, 401)]

    # Make VI workers
    workers_list = [VI_worker_class.remote(A = A,
                    successors = [[env.GetSuccessors(state, action) for action in range(A)] 
                                for state in range(batch_indx[0], batch_indx[1])],
                    rewards = env.TransitReward[batch_indx[0]:batch_indx[1], :], 
                    beta = beta, 
                    start_state = batch_indx[0], 
                    end_state = batch_indx[1]) 
                for batch_indx in batch_indxs]
    
    # Do VI computation
    error = float('inf')
    while error > epsilon:
        object_list = [workers_list[i].compute.remote(v_current) for i in range(workers_num)]
        # Wait for workers to finish

        results = ray.get(object_list)
        
        
        if(synchronous):
            sum_error = 0
            for vals, error, start_ind in  results:
                sum_error += error
                val_len = len(vals)
                v_current[start_ind:start_ind+val_len] = [vals[i][0] for i in range(val_len)]
                pi[start_ind:start_ind+val_len] = [vals[i][1] for i in range(val_len)]
            print("Error:",sum_error)
            error = sum_error
            

        if(not synchronous):
            error_list = []
            for i in range(workers_num):
                finish_id = ray.wait(object_list, num_returns = 1, timeout = None)[0][0]
                object_list.remove(finish_id)
                vals, error, start_ind = ray.get(finish_id)
                error_list.append(error)
                val_len = len(vals)
                v_current[start_ind:start_ind+val_len] = [vals[i][0] for i in range(val_len)]
                pi[start_ind:start_ind+val_len] = [vals[i][1] for i in range(val_len)]

                print("Error:",error)

            error = max(error_list)
    
    return v_current, pi
