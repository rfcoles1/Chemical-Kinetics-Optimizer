import math
import numpy as np
import matplotlib.pyplot as plot

from pylab import savefig
import os


"""
A + B --k1--> F
A + C --k2--> G
F + G --k3--> X
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
	F = r[3]
	G = r[4]
	X = r[5]
		

	k1 = r[6]
	k2 = r[7]
	k3 = r[8]

	fA = -k1*A*B - k2*A*C 
	fB = -k1*A*B
	fC = -k2*A*C
	fF = k1*A*B -k3*F*G
	fG = k2*A*C  + -k3*F*G
	fX = k3*F*G
	
	 	
	return np.array([fA,fB,fC,fF,fG,fX,k1,k2,k3], float)

def loopstatement(timelimit, currtime):
	if timelimit==-1:
		return True
	elif currtime < timelimit:
		return True
	else:
		return False

def reaction(A,B,C,k1,k2,k3,timelimit = -1, graph = 0, index = 0):

	F = 0
	G = 0
	X = 0

	r = np.array([A,B,C,F,G, X, k1,k2,k3],float)

	t = 0
	h = 0.0033

	if graph==1:
		A_values = [A]
		B_values = [B]
		C_values = [C]
		F_values = [F]
		G_values = [G]
		X_values = [X]
		tvalues = [t]


	while(loopstatement(timelimit,t)):
		t += h
	
		temp = r[2]
		r += rk4(r,t,h)
	
		#if r[2] == temp:	
		#	break
	
		if graph==1:
			A_values.append(r[0])
			B_values.append(r[1])
			C_values.append(r[2])
			F_values.append(r[3])
			G_values.append(r[4])
			X_values.append(r[5])
			tvalues.append(t)
	
	if graph==1:
		plot.figure(1)
		plot.plot(tvalues,A_values,'b', label = 'A')
		plot.plot(tvalues,B_values, 'm', label = 'B')
		plot.plot(tvalues,C_values, 'c', label = 'C')
		#plot.plot(tvalues,F_values,'g--', label = 'F')
		#plot.plot(tvalues,G_values, 'k.-', label = 'G')
		plot.plot(tvalues,X_values,'r', label = 'X')
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

	return r[5]	

def get_Nk():
	rates = 3
	return rates
