import numpy as np
import matplotlib.pyplot as plt

import Field
import Robot
import util

################################################################################
##                              simulation
################################################################################

startpos   = np.array([[900], [600], [-np.pi/4], [0], [0]]);
starterr   = np.array([[0], [0], [0], [0], [0]])

field      = Field.NewField(1200, 800, 200, 4)                                  # table
robot      = Robot.NewRobot()

field.goals= np.array([[500, 300], [700, 700], [1100, 100], [300, 500], [700, 700], [1100, 100]])

robot.d     = 90
robot.set_Ts(0.2)
robot.maxv  = 10
robot.maxw  = np.pi/32

robot.x    = startpos.copy()
robot.xhat = startpos + starterr
robot.xodo = robot.xhat.copy()
robot.s    = np.diag([5, 5, .1, 20, 20])**2;

field.xreal = robot.x.copy()                                                    # prepare record of (mis)deeds
field.xhat  = robot.xhat.copy()
field.xodo  = robot.xodo.copy()
field.s     = [2*np.sqrt(robot.s[0,0]+robot.s[1,1])]


goals       = np.ndarray.tolist(field.goals)

arrived = False
cnt = 0
print(goals)
while goals and (cnt < 2000):
    cnt = cnt +1;

    u,local_obstacle = robot.pilot(goals)

    if not local_obstacle :
        x = robot.simulate_motion(u, np.array([4, 4])).copy()

    sensor_data = robot.simulate_measure(field, np.array([4, 4]))

    if local_obstacle :
        u = sensor_data[9:11,:]
        u[u>2**15] = u[u>2**15] - 2**16

    y = robot.sensor_interpretation(sensor_data, field)

    if not (cnt%100) :                                                          #simulate regular camera updates
        c0 = np.concatenate((np.eye(3), np.zeros((3,2))), axis = 1)
        y0 = robot.x[0:3,:]
        r0 = np.array([5, 5, .1])**2                                         # we assume the camera position is the most precise possible
        robot.R = np.diag(np.concatenate((r0, np.diag(robot.R)), axis=0))
        robot.C = np.concatenate((c0, robot.C))
        y = np.concatenate((y0, y), axis=0)

    xtemp, stemp = robot.kalman_estimator(u, y)

    robot.xodo  = robot.motion_model(robot.xodo, u)                             # make a purely odometry-based estimate, for comparison

    field.xreal = np.concatenate((field.xreal,x), axis=1)
    field.xhat  = np.concatenate((field.xhat, xtemp), axis=1)
    field.xodo  = np.concatenate((field.xodo, robot.xodo), axis=1)
    field.s.append(2*np.sqrt(stemp[0,0]+stemp[1,1]))

print(cnt)
print("-----\n")

################################################################################
##                                  plot
################################################################################

field.s = np.array(field.s)

fig, ax = plt.subplots()
fig.set_size_inches(12,8)

field.plot()

plt.show()
