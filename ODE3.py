# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:13:14 2016

@author: midhununnikrishnan
"""



from scipy.integrate import ode
#import matplotlib as mpl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt

A = np.matrix([[ 1.41893582,  0.27970156],[ 0.13188878,  1.79065912]]);
#np.matrix(np.random.random((2,2)))
#A[0,0] += 1
#A[1,1] += 1
#np.random.random((2,2))
#np.matrix([[1.0,0.0],[1.0,0.0]])
#B = np.matrix(np.random.random((2,2)))
#B[0,0] += 1
#B[1,1] += 1
B = np.transpose(A)*-1.0 + np.matrix([[2.0,2.0],[2.0,2.0]])
#np.matrix([[0.0,1.0],[1.0,0.0]])
def f(t, y, arg1):
    A1 = np.dot(A,[y[1],1.0 - y[1]])
    A2 = np.dot(B,[y[0],1.0 - y[0]])
    Q1 = np.dot(A1,[y[0],1.0 - y[0]])
    Q2 = np.dot(A2,[y[1],1.0 - y[1]])
    C = [0,0,0]
    C[0] = y[2]*y[0]*(A1[0]-Q1)[0,0]
    C[1] = (1-y[2])*y[1]*(A2[0]-Q2)[0,0]
    C[2] = y[2]*(-1.0*y[2]*Q1 + y[2]*y[2]*Q1 + (1.0 - y[2])*(1.0 - y[2])*Q2)[0,0]
    return C

def f2c(Q1,Q2):
    den = (1.0/Q1) + (1.0/Q2)
    return 1.0/(den*Q1)

def f2(t,y,arg1):
    A1 = np.dot(A,[y[1],1.0 - y[1]])
    A2 = np.dot(B,[y[0],1.0 - y[0]])
    Q1 = np.dot(A1,[y[0],1.0 - y[0]])
    Q2 = np.dot(A2,[y[1],1.0 - y[1]])
    mu = f2c(Q1,Q2)
    C = [0,0]
    C[0] = mu*y[0]*(A1[0]-Q1)[0,0]
    C[1] = (1-mu)*y[1]*(A2[0]-Q2)[0,0]
    return C

def expand(xarr):
    l = np.shape(xarr)
    xarr2 = np.empty((l[0]+1,l[1]))
    xarr2[0:l[0],:] = xarr
    for i in range(l[1]):
        y = xarr[:,i]
        A1 = np.dot(A,[y[1],1.0 - y[1]])
        A2 = np.dot(B,[y[0],1.0 - y[0]])
        Q1 = np.dot(A1,[y[0],1.0 - y[0]])
        Q2 = np.dot(A2,[y[1],1.0 - y[1]])
        xarr2[l[0],i] = f2c(Q1,Q2)
    return xarr2
    
def solveODE(y0 = [0.25,0.25,0.25],t0 = 0,dt = 0.1, tmax = 1000):
    r = ode(f).set_integrator('zvode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0,t0).set_f_params(10)
    ind = 1
    xarr = np.empty((len(y0),tmax))
    xarr[:,0] = y0

    while r.successful() and ind < tmax:
        r.integrate(r.t+dt)
        xarr[:,ind] = list(r.y);
        ind += 1
    return xarr
   
def solveODE2(y0 = [0.25,0.25],t0 = 0,dt = 0.1, tmax = 1000):
    r = ode(f2).set_integrator('zvode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0,t0).set_f_params(10)
    ind = 1
    xarr = np.empty((len(y0),tmax))
    xarr[:,0] = y0

    while r.successful() and ind < tmax:
        r.integrate(r.t+dt)
        xarr[:,ind] = list(r.y);
        ind += 1
    return xarr
   
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)
timetaken = 500
initsol = solveODE([0.25,0.55,0.25],0,0.1,timetaken)
modsol = solveODE2([0.25,0.55],0,0.1,timetaken)
modsol = expand(modsol)
data = [initsol,modsol]


# Creating line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('Dynamics of Random Game')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, timetaken - 1, fargs=(data, lines),
                                   interval=5, blit=False)
plt.show()
