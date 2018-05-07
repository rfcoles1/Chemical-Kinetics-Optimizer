import math
import numpy as np
import matplotlib.pyplot as plot

from pylab import savefig
import os

"""
A + B + E --> Z + Y + E

E+A --k1--> EA
EA --k1--> E+A
EA + B --k2--> EZ + Y
EZ --k3--> E+Z
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
	Y = r[2]
	Z = r[3]
	E = r[4]

	EA = r[8]
	EZ = r[9]

	k1 = r[5]
	k2 = r[6]
	k3 = r[7]

	fA = -k1*E*A + k1*EA  
	fB = -k2*EA*B
	fY = k2*EA*B
	fZ = k3*EZ
	fE = -k1*E*A + k1*EA + k3*EZ
	fEA = k1*E*A - k1*EA - k2*EA*B	
	fEZ = k2*EA*B - k3*EZ
	 	
	return np.array([fA,fB,fY,fZ,fE,k1,k2,k3,fEA,fEZ], float)

def loopstatement(timelimit, currtime):
	if timelimit==-1:
		return True
	elif currtime < timelimit:
		return True
	else:
		return False

def reaction(A,B,E,k1,k2,k3,timelimit = -1, graph = 0, index = 0):

	Y = 0
	Z = 0

	EA = 0
	EZ = 0


	r = np.array([A,B,Y,Z,E,k1,k2,k3,EA,EZ],float)

	t = 0
	h = 0.0033

	if graph==1:
		Avalues = [A]
		Bvalues = [B]
		Yvalues = [Y]
		Zvalues = [Z]
		Evalues = [E]
		tvalues = [t]
		#EAvalues = [EA]
		#EZvalues = [EZ]

	while(loopstatement(timelimit,t)):
		t += h
	
		temp = r[2]
		r += rk4(r,t,h)
	
		if r[2] == temp:	
			break
	
		if graph==1:
			Avalues.append(r[0])
			Bvalues.append(r[1])
			Yvalues.append(r[2])
			Zvalues.append(r[3])
			Evalues.append(r[4])
			#EAvalues.append(r[8])
			#EZvalues.append(r[9])
			tvalues.append(t)
	
	if graph==1:
		plot.figure(1)
		plot.plot(tvalues,Avalues,'b', label = 'A')
		plot.plot(tvalues,Bvalues, 'c', label = 'B')
		plot.plot(tvalues,Yvalues, 'm', label = 'Y')
		plot.plot(tvalues,Zvalues,'r', label = 'Z')
		plot.plot(tvalues,Evalues, 'k', label = 'E')
		#plot.plot(tvalues,EAvalues,'b--')
		#plot.plot(tvalues,EZvalues, 'm--')
		plot.legend(loc = 2)
		plot.xlabel('Time (s)')
		plot.ylabel('Amount')
		if index > 0:
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
