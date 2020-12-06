import numpy as np
import matplotlib.pyplot as plt

import Field
import Robot
import util

################################################################################
##                              simulation
################################################################################

startpos   = np.array([[1053],  [506], [np.pi*3/4], [0], [0]]);
starterr   = np.array([[0], [0], [0], [0], [0]])

field      = Field.NewField(1200, 800, 200, 4)                                  # table
robot      = Robot.NewRobot()

field.goals= np.array([[1053,  506], [1040,  506], [1026,  506], [1013,  506],
                       [1000,  493], [ 986,  480], [ 973,  466], [ 960,  466],
                       [ 946,  466], [ 933,  466], [ 920,  466], [ 906,  453],
                       [ 893,  440], [ 880,  426], [ 866,  413], [ 853,  400],
                       [ 840,  386], [ 826,  386], [ 813,  386], [ 800,  386],
                       [ 786,  386], [ 773,  386], [ 760,  386], [ 746,  386],
                       [ 733,  386], [ 720,  386], [ 706,  386], [ 693,  386],
                       [ 680,  386], [ 666,  386], [ 653,  386], [ 640,  386],
                       [ 626,  386], [ 613,  386], [ 600,  386], [ 586,  386],
                       [ 573,  386], [ 560,  386], [ 546,  386], [ 533,  386],
                       [ 520,  386], [ 506,  386], [ 493,  386], [ 480,  386],
                       [ 466,  386], [ 453,  386], [ 440,  386], [ 426,  386],
                       [ 413,  386], [ 400,  386], [ 386,  386], [ 373,  386],
                       [ 360,  386], [ 346,  386], [ 333,  386], [ 320,  386],
                       [ 306,  386], [ 293,  386], [ 280,  386], [ 266,  386],
                       [ 253,  386], [ 240,  386], [ 226,  386], [ 213,  386],
                       [ 200,  386], [ 186,  400], [ 173,  400], [ 160,  400],
                       [ 146,  413]])

Tr = .2                                                                         # robot update timestep
Tc =  2                                                                         # camera update timestep

#robot.d     = 90
robot.set_Ts(Tr)
robot.maxv  = 10
robot.maxw  = np.pi/32

robot.x    = startpos.copy()
robot.xhat = startpos + starterr
robot.xodo = robot.xhat.copy()
robot.s    = np.diag([1, 1, .1, 20, 20])**2

field.xreal = startpos.copy()                                                   # prepare record of (mis)deeds
field.xhat  = robot.xhat.copy()
field.xodo  = robot.xodo.copy()
field.s     = [2*np.sqrt(robot.s[0,0]+robot.s[1,1])]

goals       = np.ndarray.tolist(field.goals)

robot.q = np.array([20, 20])
robot.r = np.array([6, 6])

cnt = 0
while goals and (cnt < 4000):
    cnt = cnt + 1;

    u,local_obstacle = robot.pilot(goals)

    if not local_obstacle :
        pass                                                                    # here we would launch the motion command

    sensor_data = robot.simulate_measure(field, 4*np.ones(2))

    if local_obstacle :
        u = sensor_data[9:11,:]
        u[u>2**15] = u[u>2**15] - 2**16

    y = robot.sensor_interpretation(sensor_data, field)

    if not cnt % int(Tc/Tr) :                                                   # if a camera position is available, update
        #c0 = np.concatenate((np.eye(3), np.zeros((3,2))), axis = 1)
        #y0 = robot.x[0:3,:]
        r0 = np.array([.9, .9, .1])**2                                        # we assume the camera position is the most precise
        #robot.R = np.diag(np.concatenate((r0, np.diag(robot.R)), axis=0))      # possible to make our lives simpler
        #robot.C = np.concatenate((c0, robot.C), axis=0)
        #y       = np.concatenate((y0, y), axis=0)
        robot.R = np.diag(np.concatenate((r0, robot.r), axis=0))
        robot.C = np.eye(5)
        y       = np.concatenate((robot.x[0:3,:], sensor_data[9:11,:]), axis=0)
        field.xreal = np.concatenate((field.xreal, robot.x), axis=1)

    xtemp, stemp = robot.kalman_posterior(y)
    #xtemp, stemp = robot.kalman_estimator(u, y)

    robot.xodo = robot.motion_model(robot.xodo, u)                              # make a purely odometry-based estimate, for comparison

    robot.kalman_prior(u)                                                       # guess where we will be after move

    robot.simulate_motion(u, 30*np.ones(2))                                     # the real motion mostly happens after the function call

    field.xhat  = np.concatenate((field.xhat, xtemp), axis=1)
    field.xodo  = np.concatenate((field.xodo, robot.xodo), axis=1)
    field.s.append(2*np.sqrt(stemp[0,0]+stemp[1,1]))

print("achieved goal in:",cnt)
print("-----")

################################################################################
##                                  plot
################################################################################

field.s = np.array(field.s)

fig, ax = plt.subplots()
fig.set_size_inches(12,8)

field.plot()

plt.show()
