import numpy as np

import matplotlib.pyplot as plt
from pylab import savefig
from drawnow import drawnow, figure

import os
import sys

Ri = 4

sys.path.insert(0,'./reactions')
reactionfile = 'eq' + str(Ri)
react = __import__(reactionfile)

class reaction():
    def __init__(self):
        self.num_rates = react.get_Nk()		
        self.num_actions = self.num_rates*2 + 1	
        self.rates = np.ones(self.num_rates)
        self.inputA = 100
        self.inputB = 100
        self.inputC = 100
        self.inputE = 100		

        self.timelimit = 1
        self.maxK = 2
        self.minK = 0
        self.increment = 0.01

    def runReac(self):
        currReward = react.reaction(self.inputA, self.inputB, self.inputC, self.rates[0], self.rates[1], self.rates[2], self.timelimit)
        return currReward

    def getRates(self):
	    return self.rates				

React = reaction()

total_episodes = 5000

bestk = np.zeros(React.num_rates)
bestreward = 0

for i in range(total_episodes):
    params = np.random.rand(React.num_rates)*2
    React.rates = params
    thisreward = React.runReac()
    if thisreward > bestreward:
        bestk = params
        bestreward = thisreward

    if i%500 == 0:
        print bestk, bestreward


