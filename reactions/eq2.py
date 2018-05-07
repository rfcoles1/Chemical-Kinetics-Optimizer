import math
import numpy as np
import matplotlib.pyplot as plot

from pylab import savefig
import os
"""
A + B --k1--> 2B
B + C --k2--> 2C
C --k3--> D
"""

def rk4(r,t,h):
	k1 = h*f(r,t)
	k2 = h*f(r+0.5*k1, t+0.5*h)
	k3 = h*f(r+0.5*k2, t+0.5*h)
	k4 = h*f(r+k3,t+h)
	return (k1+2*k2+2*k3+k4)/6.0

def f(r,t):
	A = r[0]
	B = r[1]
	C = r[2]
	D = r[3]

	k1 = r[4]
	k2 = r[5]
	k3 = r[6]

	fA = -k1*A*B
	fB = k1*A*B - k2*B*C
	fC = k2*B*C - k3*C
	fD = k3*C
	
	return np.array([fA,fB,fC,fD,k1,k2,k3], float)

def loopstatement(timelimit, currtime):
	if timelimit==-1:
		return True
	elif currtime < timelimit:
		return True
	else:
		return False

def reaction(A,B,C,k1,k2,k3,timelimit = -1, graph = 0, index = 0):
	D = 0
	
	r = np.array([A,B,C,D,k1,k2,k3],float)
	
	t = 0
	h = 0.0033
	if graph==1:
		Avalues = [A]
		Bvalues = [B]
		Cvalues = [C]
		Dvalues = [D]
		tvalues = [t]

	while(loopstatement(timelimit,t)):
		t += h
		
		temp = r[3]
		r += rk4(r,t,h)

		if r[3] == temp:
			break

		if graph==1:
			Avalues.append(r[0])
			Bvalues.append(r[1])
			Cvalues.append(r[2])
			Dvalues.append(r[3])
			tvalues.append(t)
	
	if graph==1:	
		plot.figure(1)
		plot.plot(tvalues,Avalues,'b', label = 'A')
		plot.plot(tvalues,Bvalues, 'c', label = 'B')
		plot.plot(tvalues,Cvalues, 'g', label = 'C')
		plot.plot(tvalues,Dvalues,'r', label = 'D')
		plot.legend(loc = 2)
		plot.xlabel('Time (s)')
		plot.ylabel('Amount')
		#plot.ylim([0,300])
		if index>0:
			plot.title('Episode %d'%(10*(int(index)/10)))
			fname = '_kin%05d.png'%(index)
			savefig(fname)
		else:
			plot.show()	
		plot.clf()

	return r[3]

def get_Nk():
	rates = 3
	return rates
