import numpy as np

side = 0

delta_x = 720 - 770                # dist next goal & robot
delta_y = 180 - 300
robot_ang = -.3
print("robot_ang", np.rad2deg(robot_ang))
ang_to_goal = np.arctan2(delta_y, delta_x) # between -pi and pi
print("ang_to_goal", np.rad2deg(ang_to_goal))
while ang_to_goal > 2*np.pi:
    ang_to_goal = ang_to_goal - 2*np.pi
while ang_to_goal < 0:
    ang_to_goal = ang_to_goal + 2*np.pi
while robot_ang > 2*np.pi:
    robot_ang = robot_ang - 2*np.pi
while robot_ang < 0:
    robot_ang = robot_ang + 2*np.pi

delta_ang = ang_to_goal - robot_ang
while delta_ang > 2*np.pi:
    delta_ang = delta_ang - 2*np.pi
while delta_ang < 0:
    delta_ang = delta_ang + 2*np.pi
if delta_ang > np.pi:
    goal_side = 1 # goal on the right
if delta_ang < np.pi:
    goal_side = 0 # goal on the left

print("side:",side)
print("goal side:",goal_side)
if side != goal_side:
    print("leave l_obs_mode")
    #robot.th.set_var("event.args", 2)
    #local_obstacle = False
