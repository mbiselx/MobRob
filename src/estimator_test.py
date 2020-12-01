import numpy as np
import matplotlib.pyplot as plt

import Field
import Robot
import util

################################################################################
##                              simulation
################################################################################

field       = Field.NewField(1200, 800, 200, 4)                                 # table

startpos    = np.array([[100], [100], [np.pi/16], [0], [0]])
starterr    = np.array([[10], [10], [0.1], [10], [10]]) * np.random.randn(5,1)

robot1      = Robot.NewRobot()                                                  # 2 robots, to make sure there's no accidental crossover
robot1.x    = startpos.copy()

robot2      = Robot.NewRobot()
robot2.xhat = startpos + starterr                                               # we have an initial error on our estimate
robot2.s    = np.diag([25, 25, .5, 20, 20])**2;


ur = np.array([[100, -50, -50, -100, -100, -50, -50, 100],                      # turn right
               [100,  50,  50,  100,  100,  50,  50, 100]])
us = np.array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],         # go straight
               [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]])
ul = np.array([[100,  50,  50,  100,  100,  50,  50, 100],                      # turn left
               [100, -50, -50, -100, -100, -50, -50, 100]])

u = np.concatenate((ur, us, us, us, us, ul, ul, ul, us, us, us, us, ul, ul, ul, ul, us, us, us, us, ur, ur, ur, ur, us, us, us, us), axis=1)

N  = u.shape[1]
x  = np.zeros((5,N))
xhat = np.zeros((5,N))

s           = [2*np.sqrt(100+100)]
x[:, 0]     = startpos.copy()[:,0]
xhat[:, 0]  = (startpos + starterr)[:,0]
sensor_data = robot1.simulate_measure(field)

for i in range(1, N):

    x[:, i] = robot1.simulate_motion(np.array([u[:, i]]).T)[:,0]                # simulate robot & environment
    sensor_data = robot1.simulate_measure(field)

    y = robot2.sensor_interpretation(sensor_data, field)                            # do robot stuff
    xtemp, stemp = robot2.kalman_estimator(np.array([u[:, i]]).T, y)

    xhat[:, i] = xtemp[:,0]                                                     # record robot stuff
    s.append(2*np.sqrt(stemp[0,0]+stemp[1,1]))

s = np.array(s)


################################################################################
##                                  plot
################################################################################

for i in range(int((field.xmax - field.xmin)/field.grid)+1) :                   # plot vertical gridlines
    plt.plot(i*field.grid*np.ones((2,1)), np.array([field.ymin, field.ymax]), '-k')
for i in range(int((field.ymax - field.ymin)/field.grid)+1) :                   # plot horizontal gridlines
    plt.plot(np.array([field.xmin, field.xmax]), i*field.grid*np.ones((2,1)), '-k')

cos, sin = np.cos(xhat[2,:]), np.sin(xhat[2,:])
plt.plot(xhat[0,:] - s*sin, xhat[1,:] + s*cos, '--c')                           # plot uncertainty
plt.plot(xhat[0,:] + s*sin, xhat[1,:] - s*cos, '--c')

plt.plot( x[0,:], x[1,:], '-g', linewidth=3)                                    # plot real trajectory
plt.plot( xhat[0,:], xhat[1,:], '-r')                                           # plot estimated trajectory

start=[]
R = util.rot_mat2D(startpos[2][0])
for i in range(len(robot1.outline)):
    start.append(R @ robot1.outline[i,:])
start = startpos[0:2] + np.array(start).T
plt.plot(start[0,:], start[1,:], '-b')
for sensor in robot1.proxv:
    sp = R @ sensor.pos + startpos[0:2]
    plt.scatter(sp[0], sp[1], marker="o", color='b')

plt.axis("equal")
plt.show()
