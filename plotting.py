import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
import os
import sys

Ri = 4
colourplot = 1

sys.path.insert(0,'/.reactions')
reactionfile = 'eq' + str(Ri)
react = __import__(reactionfile)


Results = np.loadtxt('OptimizeResults.dat', float)
Product = Results[:,0]
Rates = Results[:,1:]
num_rates = np.shape(Rates)[1]
total_frames = np.shape(Rates)[0]

plt.figure(1)
plt.xlabel("Episode")
plt.ylabel("Product")
plt.plot(Product)

plt.figure(2)
for j in range(num_rates):
    plt.plot(Rates[j], label ='k%d'%(j+1))	
plt.xlabel("Episode")
plt.ylabel("Reaction Rates")
plt.legend(loc = 2)

k_values = np.arange(0,2.1,0.1)
n = len(k_values)
results = np.zeros((n,n))
maxresult = 0
max1 = 0
max2 = 0
max3 = 0

#obtains obtimal results 
for i in range(n):
    for j in range(n):
        for k in range(n):
            curr = react.reaction(100,100,100,k_values[i], k_values[j],k_values[k],1)
            if curr > maxresult:
                maxresult = curr
                max1 = k_values[i]
                max2 = k_values[j]
                max3 = k_values[k]			

            if colourplot == 1:
                if Ri == 1: 
                    if k_values[j] == 0:
                        results[i,k] = curr
                if Ri == 2:
                    if k_values[i] == 2:
                        results[j,k] = curr
                if Ri == 3:
                    if k_values[k] == 2:
                        results[i,j] = curr
                if Ri == 4:
                    if k_values[k] == 2:
                        results[i,j] = curr

plt.figure(1)
plt.plot([0,total_frames],[maxresult, maxresult], 'b--')
savefig('ProductGraph')

plt.figure(2)
plt.plot([0,total_frames],[max1,max1], 'b--')
plt.plot([0,total_frames], [max2,max2], 'y--')
plt.plot([0,total_frames], [max3,max3], 'g--')
savefig('RatesGraph')

if colourplot == 1:
    plt.figure(3)
    plt.contourf(k_values, k_values, results,100)
    plt.xlabel('k1')
    plt.ylabel('k3')
    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.colorbar()
    plt.plot(Rates[0], Rates[2], lw = 2.5,c = 'k')
    savefig('ContourGraph')

