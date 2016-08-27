# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:10:49 2016

@author: midhununnikrishnan
"""


from scipy.integrate import ode
#import matplotlib as mpl
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt

def f(t, y, arg1):
    return [2*y[1]*y[1] - 2*y[2]*y[2],y[2],-y[1]]

def jac(t,y,arg1):
    return [[0,4.0*y[1],-4.0*y[2]],[0,0,1],[0,-1,0]]
    
def solveODE(y0 = [0.0,0.0,1.0],t0 = 0,dt = 0.01, tmax = 1000):
    r = ode(f, jac).set_integrator('zvode', method='bdf', with_jacobian=True)
    r.set_initial_value(y0,t0).set_f_params(10).set_jac_params(10)
    ind = 1
    xarr = np.empty((3,tmax))
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

data = [solveODE([0.0,0.0,1.0]),solveODE([0.1,0.1,0.9])]

# Creating line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 999, fargs=(data, lines),
                                   interval=5, blit=False)
plt.show()

#ax.plot(list(map(lambda x: x[0],xarr)),list(map(lambda x: x[1],xarr)),list(map(lambda x: x[2],xarr)), label='parametric curve')
#fig = plt.figure()
#ax = p3.Axes3D(fig)
