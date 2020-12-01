import numpy as np
import matplotlib.pyplot as plt

import Field
import Robot
import util

################################################################################
##                              simulation
################################################################################

startpos   = np.array([[100], [100], [np.pi/4], [0], [0]]);
starterr   = np.array([[0], [0], [0], [0], [0]])

field      = Field.NewField(1200, 800, 200, 4)                                  # table
robot      = Robot.NewRobot()

field.goals= [[500, 300], [700, 700], [1100, 100], [300, 500], [700, 700], [1100, 100]]

robot.x    = startpos.copy()
robot.xhat = startpos + starterr
robot.xodo = robot.xhat.copy()
robot.s    = np.diag([25, 25, .5, 20, 20])**2;

field.xreal = robot.x.copy()                                                    # prepare record of (mis)deeds
field.xhat  = robot.xhat.copy()
field.xodo  = robot.xodo.copy()
field.s     = [2*np.sqrt(50+50)]


sensor_data = robot.simulate_measure(field)
goals       = field.goals.copy()

arrived = False
cnt = 0
print(goals)
while goals and (cnt < 2000):
    cnt = cnt +1;
    prev_sensor_data = sensor_data

    u = robot.pilot(goals)

    x = robot.simulate_motion(u, np.array([4, 4])).copy()
    sensor_data = robot.simulate_measure(field, np.array([4, 4]))

    y = robot.sensor_interpretation(sensor_data, field)
    xtemp, stemp = robot.kalman_estimator(u, y)

    dist = np.linalg.norm(np.array([field.xhat[0:2,-1]]).T-xtemp[0:2])
    if dist > 10 :
        print("Help, I'm lost. Cheating")
        xtemp = robot.x + np.array([[2], [2], [0.1], [10], [10]]) * np.random.randn(5,1);
        stemp = np.diag([10, 10, .2, 10, 10])**2;
        robot.xhat = xtemp.copy()
        robot.s    = stemp.copy()

    robot.xodo = robot.motion_model(robot.xodo, u) #make a purely odometry-based estimate, for comparison

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
