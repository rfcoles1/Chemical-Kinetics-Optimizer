import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os 
import sys

Ri = 4

sys.path.insert(0,'./reactions')
reactionfile = 'eq' + str(Ri)
react = __import__(reactionfile)

class ChemKinEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_rates = react.get_Nk()
        self.num_actions = self.num_rates*2 + 1
        self.rates = np.ones(self.num_rates)
        self.inputA = 100
        self.inputB = 100
        self.inputC = 100
        
        self.timelimit = 1
        self.maxK = 2
        self.minK = 0
        self.increment = 0.01


    def step(self, action):
        currProd = react.reaction(self.inputA, self.inputB, self.inputC, self.rates[0]. self.rates[1], self.rates[2], self.timelimit)

        take_action(action)

        nextProd = react.reaction(self.inputA, self.inputB, self.inputC, self.rates[0]. self.rates[1], self.rates[2], self.timelimit)

        reward = get_reward(currProd, nextProd):
        
        return nextProd, reward

    def take_action(self,action):
        for i in range(0, self.num_rates):
            if (action == i) & (self.rates[i] < self.maxK):
                self.rates[i] += self.increment
            if (action == (i + self.num_rates)) & (self.rates[i] > self.minK):
                self.rates[i] -= self.increment

    def get_reward(self, curr, nxt):
        if (nxt - curr) <= 0:
            return -1
        else:
            return 1

    def get_rates(self):
        return self.rates

    def reset(self):
        
