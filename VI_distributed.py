import argparse
import ray
import time
import numpy as np

ray.shutdown()
ray.init()

# A : Action Space 
# S : State Space 



@ray.remote
class VI_worker_class(object):
    def __init__(self, list_of_actions, tran_dict, reward_dict, beta, list_of_states):
        self.list_of_states = list_of_states
        self.list_of_actions = list_of_actions
        self.tran_dict = tran_dict
        self.reward_dict = reward_dict
        self.beta = beta
        self.list_of_states = list_of_states
        
        
    def compute(self, values):
        return_vals = {}
        for state in self.list_of_states:
            return_vals[state] = float('-inf')

        max_error = 0
        
        for state in self.list_of_states:
            for action in self.tran_dict[state]:
                expected_val = 0
                successors = self.tran_dict[state][action].keys()
                for next_state in successors:
                    try:
                        expected_val += self.tran_dict[state][action][next_state] * values[next_state]
                    except:
                        expected_val += self.tran_dict[state][action][next_state] * 0
                new_val = self.reward_dict[state][action] + self.beta*expected_val
                if new_val > return_vals[state]:
                    return_vals[state] = new_val
            error = abs(return_vals[state] - values[state])
            if error > max_error:
                max_error = error
        return return_vals, max_error

def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4):
    """
    :param env: Given environment/MDP must have following functions:
                GetStateSpace() -> returns the list of all states in the MDP
                GetActionSpace() -> returns all possible actions in the MDP
                TransitReward(s,a)-> returns rewards for transition (state, action)
                GetSuccessors(s,a) -> returns list of tuples each containing probable next_state and the correspondin probability
    :param beta:
    :param epsilon:
    :param workers_num:
    :return:
    """
    # Make VI server
    S = env.GetStateSpace()
    A = env.GetActionSpace()

    list_of_states = list(range(S))
    list_of_actions = list(range(A))

    batch_indxs = [(int(i*(S/workers_num)), int((i+1)*(S/workers_num))) for i in range(workers_num)]
    list_of_list_of_states = [range(batch_indx[0],batch_indx[1]) for batch_indx in batch_indxs]

    v_current = {s:0 for s in list_of_states}
    pi = {s:0 for s in list_of_states}

    reward_dict = {}
    tran_dict = {}
    for s in list_of_states:
        reward_dict[s] = {}
        tran_dict[s] = {}
        for a in list_of_actions:
            reward_dict[s][a]=env.TransitReward(s,a)

            tran_dict[s][a]={}
            successors = env.GetSuccessors(s,a)
            for ns, p in successors:
                tran_dict[s][a][ns] = p

    # Make VI workers
    workers_list = [VI_worker_class.remote(list_of_actions=list_of_actions,
                                           tran_dict=tran_dict,
                                           reward_dict=reward_dict,
                                           beta=beta,
                                           list_of_states=list_of_states)
                    for list_of_states in list_of_list_of_states]
    
    # Do VI computation
    error = float('inf')
    while error > epsilon:
        object_list = [workers_list[i].compute.remote(v_current) for i in range(workers_num)]
        # Wait for workers to finish

        results = ray.get(object_list)
        
        error_list = []
        for i in range(workers_num):
            finish_id = ray.wait(object_list, num_returns = 1, timeout = None)[0][0]
            object_list.remove(finish_id)
            v_new, error = ray.get(finish_id)
            error_list.append(error)
            v_current.update(v_new)

            print("Error:",error)

        error = max(error_list)
    
    return list(v_current.values()), list(pi.values())

def fast_value_iteration_v2(list_of_states, list_of_actions, reward_dict, tran_dict, beta = 0.999, epsilon = 0.01, workers_num = 4,verbose = True,keep_time = False):
    if(keep_time):
        start_time = time.time()
    # Make VI server
    all_rewards = [reward_dict[s][a] for s in reward_dict.keys() for a in reward_dict[s] ]
    
    print("Number of States",len(list_of_states))
    print("Number of transitions", len(tran_dict.keys())*2)
    print("Number of reward transitions", len(all_rewards))
    print("Number of End States", len(all_rewards)- sum(all_rewards))

    S = len(list_of_states)

    batch_indxs = [(int(i*(S/workers_num)), int((i+1)*(S/workers_num))) for i in range(workers_num)]
    list_of_list_of_states = [list_of_states[batch_indx[0]:batch_indx[1]] for batch_indx in batch_indxs]

    v_current = {s:0 for s in list_of_states}
    pi = {s:0 for s in list_of_states}

    # Make VI workers
    workers_list = [VI_worker_class.remote(list_of_actions = list_of_actions,
                                            tran_dict = tran_dict,
                                            reward_dict = reward_dict, 
                                            beta = beta, 
                                            list_of_states = list_of_states) 
                   for list_of_states in list_of_list_of_states]
    
    # Do VI computation
    error = float('inf')
    while error > epsilon:
        object_list = [workers_list[i].compute.remote(v_current) for i in range(workers_num)]
        # Wait for workers to finish

        results = ray.get(object_list)
        
        error_list = []
        for i in range(workers_num):
            finish_id = ray.wait(object_list, num_returns = 1, timeout = None)[0][0]
            object_list.remove(finish_id)
            v_new, error = ray.get(finish_id)
            error_list.append(error)
            v_current.update(v_new)
            if(verbose):
                print("Error:",error)

        error = max(error_list)

    pi = get_pi_from_value(v_current,list_of_actions,tran_dict,reward_dict,beta)
    
    if(keep_time):
        print("Time take for Value Iteration",time.time()-start_time)

    return v_current, pi



def get_pi_from_value(V, list_of_actions, tran_dict, reward_dict, beta):
    r_v = {}
    pi = {}
    for s in V:
        r_v[s] = float('-inf')

    for s in V:
        action_vals = []
        for a in tran_dict[s]:
            expected_val = 0
            successors = tran_dict[s][a].keys()
            for ns in successors:
                try:
                    expected_val += tran_dict[s][a][ns] * V[ns]
                except:
                    expected_val += tran_dict[s][a][ns] * 0
            new_val = reward_dict[s][a] + beta*expected_val
            action_vals.append(new_val)
        pi[s] = list_of_actions[np.argmax(np.array(action_vals))]

    return pi