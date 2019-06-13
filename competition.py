import time
from copy import deepcopy
# import matplotlib.pyplot as plt
from random import randint, choice
import pickle
from statistics import mean
import torch

# from frozen_lake import *

# ray.shutdown()
# ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)




import ray
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice

import pickle
from statistics import mean

import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision = 2)
TransitionProb = [0.7, 0.1, 0.1, 0.1]
def generate_row(length, h_prob):
    row = np.random.choice(2, length, p=[1.0 - h_prob, h_prob])
    row = ''.join(list(map(lambda z: 'F' if z == 0 else 'H', row)))
    return row


def generate_map(shape):
    """

    :param shape: Width x Height
    :return: List of text based map
    """
    h_prob = 0.1
    grid_map = []

    for h in range(shape[1]):

        if h == 0:
            row = 'SF'
            row += generate_row(shape[0] - 2, h_prob)
        elif h == 1:
            row = 'FF'
            row += generate_row(shape[0] - 2, h_prob)

        elif h == shape[1] - 1:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FG'
        elif h == shape[1] - 2:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FF'
        else:
            row = generate_row(shape[0], h_prob)

        grid_map.append(row)
        del row

    return grid_map



MAPS = {
    
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "16x16": [
        "SFFFFFFFFHFFFFHF",
        "FFFFFFFFFFFFFHFF",
        "FFFHFFFFHFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFHHFFFFFFFHFFFH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFHFFF",
        "FFFFFHFFFFFFFFFH",
        "FFFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFHFFFFHF",
        "FFFFFFFFFFHFFFFF",
        "FFFHFFFFFFFFFFFG",
    ],
    
    "32x32": [
        'SFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFH',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFH',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFH',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFH',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFG',
    ]
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        rew_hole = -1000
        rew_goal = 1000
        rew_step = -1
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.TransitProb = np.zeros((nA, nS + 1, nS + 1))
        self.TransitReward = np.zeros((nS + 1, nA))
        
        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'H':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_hole
                    elif letter in b'G':
                        li.append((1.0, s, 0, True))
                        self.TransitProb[a, s, nS] = 1.0
                        self.TransitReward[s, a] = rew_goal
                    else:
                        if is_slippery:
                            #for b in [(a-1)%4, a, (a+1)%4]:
                            for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransitionProb):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                #rew = float(newletter == b'G')
                                #li.append((1.0/10.0, newstate, rew, done))
                                if newletter == b'G':
                                    rew = rew_goal
                                elif newletter == b'H':
                                    rew = rew_hole
                                else:
                                    rew = rew_step
                                li.append((p, newstate, rew, done))
                                self.TransitProb[a, s, newstate] += p
                                self.TransitReward[s, a] = rew_step
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    
    def GetSuccessors(self, s, a):
        next_states = np.nonzero(self.TransitProb[a, s, :])
        probs = self.TransitProb[a, s, next_states]
        return [(s,p) for s,p in zip(next_states[0], probs[0])]
    
    def GetTransitionProb(self, s, a, ns):
        return self.TransitProb[a, s, ns]
    
    def GetReward(self, s, a):
        return self.TransitReward[s, a]
    
    def GetStateSpace(self):
        return self.TransitProb.shape[1]
    
    def GetActionSpace(self):
        return self.TransitProb.shape[0]

    
    
def evaluate_policy(env, policy, trials = 1000):
    total_reward = 0
#     epoch = 10
    for _ in range(trials):
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward
    return total_reward / trials



def evaluate_policy_discounted(env, policy, discount_factor, trials = 1000):
    epoch = 10
    reward_list = []
    
    for _ in range(trials):
        total_reward = 0
        trial_count = 0 
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += (discount_factor**trial_count)*reward
            trial_count+=1
        reward_list.append(total_reward)   
    
    return mean(reward_list)



def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np  = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor = beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size,map_size)))
    
    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))
    
def save_and_print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np  = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor = beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size,map_size)))
    plt.imsave("latest_fig.png",np.array(v_np[:-1]).reshape((map_size,map_size)),dpi=400)
    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))

def save_results(v,map_size):
    v_np = np.array(v)
    plt.imsave("latest_fig.png",np.array(v_np[:-1]).reshape((map_size,map_size)),dpi=400)








############################################################################################

def fast_value_iteration(env, beta = 0.999, epsilon = 0.01, workers_num = 4):
    init_start_time = time.time()

    def get_list_of_stateSet(state_size, splits=4):
        return torch.chunk(torch.tensor(range(state_size)), splits)


    def get_list_of_nextStateSet(list_of_stateSet, transition_prob_full):
        def get_next_states(s, a):
            return torch.flatten(torch.nonzero(transition_prob_full[a, s, :])).numpy().tolist()

        state_nextStateSet = []

        for state_set in list_of_stateSet:

            n_s_new = torch.nonzero(transition_prob_full[0:4, state_set[0]:state_set[-1] + 1, :])[:, -1]

            n_s = torch.sort(torch.unique(torch.cat((n_s_new, state_set))))[0].numpy().tolist()

            subset_curr = 0
            s_ns_map_new = deepcopy(n_s)
            SSSS = len(state_set)
            for i in range(len(n_s)):
                if (s_ns_map_new[i] == state_set[subset_curr]):
                    s_ns_map_new[i] = True
                    subset_curr += 1
                    subset_curr = min(subset_curr, SSSS - 1)
                else:
                    s_ns_map_new[i] = False

            s_ns_map = torch.tensor(s_ns_map_new, dtype=torch.uint8)
            state_nextStateSet.append((state_set, n_s, s_ns_map))

        return state_nextStateSet



    A = env.GetActionSpace()
    stateSize = env.GetStateSpace()

    print(time.time() - init_start_time)
    # init_start_time = time.time()
    start_time = time.time()
    #############################################################################################

    v_full = torch.tensor([0] * stateSize, dtype=torch.float64).reshape(1, -1)
    v_new_full = torch.tensor([0] * stateSize, dtype=torch.float64).reshape(1, -1)
    transition_prob_full = torch.tensor(env.TransitProb, dtype=torch.float64)
    state_action_reward_full = torch.tensor(env.TransitReward.T, dtype=torch.float64)
    beta = 0.999

    no_of_splits = int(sys.argv[2])

    A, S, _ = transition_prob_full.shape

    list_of_state_set = get_list_of_stateSet(S, splits=no_of_splits)
    state_nextState_set = get_list_of_nextStateSet(list_of_state_set, transition_prob_full)

    end_time = time.time()
    print("elapsed:{}".format(end_time - start_time))
    start_time = time.time()

    list_of_transition_sets = [(transition_prob_full[:, n_s, :])[:, :, n_s] for _, n_s, _ in state_nextState_set]
    list_of_value_sets = [v_full[:, n_s] for _, n_s, _ in state_nextState_set]
    list_of_new_value_sets = [v_new_full[:, n_s] for _, n_s, _ in state_nextState_set]
    list_of_state_action_reward = [state_action_reward_full[:, n_s] for _, n_s, _ in state_nextState_set]

    end_time = time.time()
    print("elapsed:{}".format(end_time - start_time))
    start_time = time.time()

    data_val = []


    def merge_values(data, S):
        stateSize = S
        v = torch.tensor([0] * stateSize, dtype=torch.float64).reshape(1, -1)
        for v_new_split, s_ns_set in data:
            v[:, s_ns_set[0]] = v_new_split[s_ns_set[2]]
        return v


    bellman_error = float('inf')
    init_start = time.time()

    while (bellman_error > 0.01):
        start_time = time.time()
        data_val = []
        for i in range(no_of_splits):
    # 		start_time = time.time()

            v = list_of_value_sets[i]
            v_new = list_of_new_value_sets[i]
            transition_prob = list_of_transition_sets[i]
            state_action_reward = list_of_state_action_reward[i]

            expected_value = torch.sum(transition_prob * v, dim=-1)
            approximated_value = state_action_reward + beta * expected_value
            v_new, pi_new = torch.max(approximated_value, dim=0)

            data_val.append((v_new, state_nextState_set[i]))
    # 		end_time = time.time()

    # 		print("elapsed 1-2:{}".format(end_time - start_time))

        v_new_full = merge_values(data_val, S)
        #     print(v_new_full)
        bellman_error = torch.max(torch.abs(v_new_full - v_full))
        v_full = v_new_full

        end_time = time.time()
        list_of_value_sets = [v_new_full[:, n_s] for _, n_s, _ in state_nextState_set]



        print(
            "Elapsed: {} iteration_time: {}, Error {}".format(end_time - init_start, end_time - start_time, bellman_error))

    end_time = time.time()
    print("elapsed:{}".format(end_time - start_time))
    print("total elapsed:{}".format(end_time - init_start_time))

    expected_value = torch.sum(transition_prob_full * v_full, dim=-1)
    approximated_value = state_action_reward_full + beta * expected_value
    v_new, pi_new = torch.max(approximated_value, dim=0)

    return v_new.numpy(), pi_new.numpy()

if __name__ == "__main__":
    curr_size = int(sys.argv[1])
    MAP = (generate_map((curr_size, curr_size)), curr_size)
    # MAP = (MAPS["32x32"], 32)
    map_size = MAP[1]
    run_time = {}

    env = FrozenLakeEnv(desc=MAP[0], is_slippery=True)

    print(fast_value_iteration(env, beta=0.999, epsilon=0.01, workers_num=4))

