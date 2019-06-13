import ray
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice
import pickle
import numpy as np

ray.shutdown()
ray.init()

@ray.remote
class error_server(object):
    def __init__(self,size,threshold):
        self.state_count = size
        self.threshold = threshold
        self.V_Pi_futures = []
        self.V_values = []
        self.Pi_values = []
        self.iter = 0
    
    def next_futures(self, futures):
        self.V_Pi_futures.append(futures)
        
    def iterate_error(self):
        if len(self.V_Pi_futures) < self.iter + 1 :
            raise Exception("Futures not yet given for iteration " + str(self.iter))
        
        values = [0]*self.state_count
        pis = [0]*self.state_count
        i = 0
        for f in self.V_Pi_futures[self.iter]:
            val_list, pi_list = ray.get(f)
            s = len(val_list)
            values[i:i+s] = val_list
            pis[i:i+s] = pi_list
            i += s
        
        self.V_values.append(values);
        self.Pi_values.append(pis);
        
        if self.iter == 0:
            self.iter += 1
            return (float('inf'),None)
        
        max_error = 0
        for i in range(self.state_count):
            error = abs(self.V_values[self.iter][i] - self.V_values[self.iter-1][i])
            if error > max_error:
                max_error = error
        
        self.V_values[self.iter-1] = []
        self.iter += 1
        if max_error > self.threshold:
            return (max_error, None)
        else:
            return (max_error,(values,pis))


@ray.remote
class state_server(object):
    
    def __init__(self,env,beta,epsilon,start,end):
        self.start = start
        self.end = end
        self.action_count = env.GetActionSpace()
        self.state_count = env.GetStateSpace()
        self.successors = [env.GetSuccessors(state,act) for state in range(start,end) for act in range(self.action_count)]
        self.rewards = [env.GetReward(state,act) for state in range(start,end) for act in range(self.action_count)]
        self.beta = beta
        self.epsilon = epsilon
        self.iter = 0
    
    def next_value_policy(self,futures):
        
        value_lists = []
        count = 0
        for f in futures:
            v,p = ray.get(f)
            count += len(v)
            value_lists.append(v)
        prev_V = [None]*count
        
        i = 0
        for v in value_lists:
            s = len(v)
            prev_V[i:i+s] = v
            i += s
        
        vals = [0]*(self.end-self.start)
        pis = [0]*(self.end-self.start)
        result = [None]*(self.end-self.start)
        s_a_itr = 0
        #print(range(start_state,end_state))
        for update_state in range(self.start,self.end):
            max_v = float('-inf')
            max_a = 0
            for action in range(self.action_count):
                succ = self.successors[s_a_itr]
                #print(succ)
                v_acc = self.rewards[s_a_itr]
                #print("prev_V: " +str(prev_V))
                for state_prime, trans_prob in succ:
                    local_v = prev_V[state_prime]
                    v_acc += local_v * trans_prob
                if v_acc > max_v :
                    max_v = v_acc
                    max_a = action
                s_a_itr += 1
            vals[update_state-self.start] = max_v
            pis [update_state-self.start] = max_a
        return vals, pis

                    
def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    A = env.GetActionSpace()
    #print([env.GetSuccessors(state,act) for state in range(0,15) for act in range(A)])
    #print([env.GetReward(state,act) for state in range(0,15) for act in range(A)])
    s_servers = [None]* workers_num
    for w in range(workers_num):
        start = int(float(S*w)/workers_num)
        end = int(float(S*(w+1))/workers_num)
        s_servers[w] = state_server.remote(env,beta,epsilon,start,end)
        
    e_server = error_server.remote(S,epsilon)
    
    futures = [None] * workers_num
    for w in range(workers_num):
        start = int(float(S*w)/workers_num)
        end = int(float(S*(w+1))/workers_num)
        count = end - start
        futures[w] = ray.put(([0]*count,[0]*count))
    #print("Set up workers")
    buffer_count = 30
    iteration = 0
    error = float('inf')
    while error > epsilon:
        #print("Test")
        last_futures = futures
        futures = [None]*workers_num
        for w in range(workers_num):
            futures[w] = s_servers[w].next_value_policy.remote(last_futures)
            
        e_server.next_futures.remote(futures)
        
        if iteration >= buffer_count :
            #print("Grabbing Error "+str(iteration-buffer_count))
            error, v_p = ray.get(e_server.iterate_error.remote())
            #print(error)
            if v_p is not None:
                (v,p) = v_p
                return v,p
        iteration += 1
            
    return v, pi