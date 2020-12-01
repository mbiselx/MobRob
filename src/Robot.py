import numpy as np
import util
import math

from Thymio import Thymio


class IRSensor:
    def __init__ (self, position, orientation):
        """ initialize a basic sensor object """
        self.pos   = np.array([position]).T
        self.alpha = orientation


class NewRobot:
    def __init__(self, x=np.zeros((5,1)), xhat=np.zeros((5,1)), s=np.eye(5), r=[2.5, 2.5], q=[2.5, 2.5], thymio=[]):
        """ Initialize the robot with standard values """

        # geometry
        self.d         = 94                                                     # [mm] distance between wheels
        self.c         = 0.4166666666666666                                     # thymio speed [th] to [mm/s]
        self.outline = np.array([[-30, -55],                                    # [mm] thymio outline, reference point at center
                                 [-30,  55],
                                 [ 55,  55],
                                 [ 63,  47],
                                 [ 74,  25],
                                 [ 80,   0],
                                 [ 75, -24],
                                 [ 64, -46],
                                 [ 55, -55],
                                 [-30, -55]])
        self.proxh = [IRSensor([ 64,  46],  0.523598),                          # horizontal proximity sensors
                      IRSensor([ 75,  24],  0.261799),
                      IRSensor([ 80,   0],  0.0),
                      IRSensor([ 74, -25], -0.261799),
                      IRSensor([ 63, -47], -0.523598),
                      IRSensor([-30,  30], -np.pi),
                      IRSensor([-30, -30], -np.pi)]
        self.proxv = [IRSensor([ 72,  12], -np.pi/2),                           # vertical proximity sensors
                      IRSensor([ 72, -12], -np.pi/2)]


        # state
        self.x    = np.array(x)                                                 # true position (x, y, phi, vr, vl) [mm, mm, rad, th, th]
        self.xhat = np.array(xhat)
        self.xodo = np.array(xhat)                                              # estimated position
        self.s    = np.array(s)                                                 # estimated variance

        # model
        self.Ts   = 0.1                                                         # [s]   sample time
        self.maxv = 15;                                                         # [mm/s] maximum speed allowed
        self.maxw = np.pi/32                                                    # [rad/s] maximum ancular velocity
        self.B    = self.Ts*self.c*np.array([[.5,  .5],[1/self.d, -1/self.d]])  # partial input transform
        self.C    = np.concatenate((np.zeros((2,3)), np.eye(2)), axis = 1)      # output matrix
        self.r    = np.array(r)                                                 # measurement variance
        self.R    = np.diag(r)                                                  # measurement covariance matrix
        self.q    = np.array(q)                                                 # process (motion) variance

        # actual thymio
        self.th = thymio

    def set_Ts(self, Ts):
        """ set sample timestep """
        self.Ts   = Ts
        self.B    = self.Ts*self.c*np.array([[.5,  .5],[1/self.d, -1/self.d]])  # partial input transform


    ###### Robot modelization
    def motion_model(self, x, u):
        """ Model of the non-holonomic motion model
            takes motor speeds as input and outputs a position estimate """

        dpol = self.B @ u
        x[3:5] = u                                                              # vr, vl

        mm = 2;                                                                 # select motion model
        if mm == 0 :                                                            # model 0) : curved tranlation
            dpol[1] = util.avoid_singularity(dpol[1])                           # avoid singularity
            x[0] = x[0] + dpol[0]/dpol[1] * (np.sin(x[2]+dpol[1]) - np.sin(x[2]))   # x
            x[1] = x[1] + dpol[0]/dpol[1] * (np.cos(x[2]) - np.cos(x[2]+dpol[1]))   # y
            x[2] = util.limit_pi(x[2] + dpol[1]);                               # phi
        if mm == 1 :                                                            # model 1) : rotate then translate
            x[2] = util.limit_pi(x[2] + dpol[1]);
            x[0] = x[0] + dpol[0]*np.cos(x[2]);
            x[1] = x[1] + dpol[0]*np.sin(x[2]);
        if mm == 2 :                                                            # model 2) : translate then rotate
            x[0] = x[0] + dpol[0]*np.cos(x[2]);
            x[1] = x[1] + dpol[0]*np.sin(x[2]);
            x[2] = util.limit_pi(x[2] + dpol[1]);

        return x

    def motion_model_jacobian(self, x, u):
        """ Jacobian of the motion model. Is dependent on position and motor speeds
            outputs a numpy matrix """

        dpol = self.B @ u
        G = np.eye(5);

        mm = 2;                                                                 # select motion model
        if mm == 0 :                                                            # model 0) : curved tranlation
            dpol[1] = util.avoid_singularity(dpol[1])                           # avoid singularity
            c = dpol[0]/dpol[1];
            G[0, 2] = c * (np.cos(x[2]+dpol[1]) - np.cos(x[2]))
            G[1, 2] = c * (np.sin(x[2]+dpol[1]) - np.sin(x[2]))
            G[0, 3] = (self.B[0,0] - self.B[1,0]*c) * (np.sin(x[2]+dpol[1]) - np.sin(x[2]))/dpol[1] + self.B[1,0]*c*np.cos(x[2]+dpol[1]);
            G[1, 3] = (self.B[0,0] - self.B[1,0]*c) * (np.cos(x[2]) - np.cos(x[2]+dpol[1]))/dpol[1] + self.B[1,0]*c*np.sin(x[2]+dpol[1]);
            G[2, 3] =  self.B[1,0]
            G[0, 4] = (self.B[0,1] - self.B[1,1]*c) * (np.sin(x[2]+dpol[1]) - np.sin(x[2]))/dpol[1] + self.B[1,1]*c*np.cos(x[2]+dpol[1]);
            G[1, 4] = (self.B[0,1] - self.B[1,1]*c) * (np.cos(x[2]) - np.cos(x[2]+dpol[1]))/dpol[1] + self.B[1,1]*c*np.sin(x[2]+dpol[1]);
            G[2, 4] =  self.B[1,1]
        if mm == 1 :                                                            # model 1) : rotate then translate
            G[0, 2] = -dpol[0]*np.sin(x[2]+dpol[1]);
            G[1, 2] =  dpol[0]*np.cos(x[2]+dpol[1]);
            G[0, 3] = self.B[0,0]*np.cos(x[2]+dpol[1]) - self.B[1,0]*dpol[0]*np.sin(x[2]+dpol[1]);
            G[1, 3] = self.B[0,0]*np.sin(x[2]+dpol[1]) + self.B[1,0]*dpol[0]*np.cos(x[2]+dpol[1]);
            G[2, 3] = self.B[1,0];
            G[0, 4] = self.B[0,1]*np.cos(x[2]+dpol[1]) - self.B[1,1]*dpol[0]*np.sin(x[2]+dpol[1]);
            G[1, 4] = self.B[0,1]*np.sin(x[2]+dpol[1]) + self.B[1,1]*dpol[0]*np.cos(x[2]+dpol[1]);
            G[2, 4] = self.B[1,1];
        if mm == 2 :                                                            # model 2) : translate then rotate
            G[0, 2] = -dpol[0]*np.sin(x[2]);
            G[1, 2] =  dpol[0]*np.cos(x[2]);
            G[0, 3] = self.B[0,0]*np.cos(x[2]);
            G[1, 3] = self.B[0,0]*np.sin(x[2]);
            G[2, 3] = self.B[1,0];
            G[0, 4] = self.B[0,1]*np.cos(x[2]);
            G[1, 4] = self.B[0,1]*np.sin(x[2]);
            G[2, 4] = self.B[1,1];

        return G

    def motion_model_covariance(self, G):
        """ model of the covariance propagation through the motion model
            outputs a numpy matrix """

        Qx = np.zeros((5,5))
        Qx[3:5, 3:5] = np.diag(self.q)
        Q = G @ Qx @ G.T;                                                       # covariance propagation estimaion in a non-linear system
        return Q

    def motion_model_inverse(self, x, xp):
        """ inverse motion model, for use in simplistic controller
            takes a goal position (xp) and a current position (x) as input and
            outputs the motor controls necessary to reach it
            The outputs are capped to avoid explosion """

        dpol = np.zeros((2,1))

        dx = xp[0:2] - x[0:2]

        dpol[1] = np.arctan2(dx[1], dx[0]) - x[2]
        dpol[1] = util.limit_pi(dpol[1])                                        # make dphi [-pi:pi]
        dpol[1] = util.avoid_singularity(dpol[1])                               # avoid singularity
        dpol[1] = util.cap_abs(dpol[1], self.maxw)                              # cap rotation so we don't spin on the spot too much (we lose a lot of precision that way)

        dpol[0] = np.abs(dpol[1]) * np.sqrt( 0.5 * (dx.T @ dx) / (1-np.cos(dpol[1])))
        dpol[0] = util.cap_abs(dpol[0], self.maxv*self.Ts)                      # cap speed

        u = np.linalg.inv(self.B) @ dpol
        return u

    def sensor_interpretation(self, sensor_data, field):
        """ interprets the raw sensor values into robot states
            takes current and previous sensor values as input, as well as
            static information about the world, and
            prepares the robot for appliction of the extended Kalman filter"""

        C = np.concatenate((np.zeros((2,3)), np.eye(2)), axis = 1)
        r = self.r
        y = sensor_data[9:11].copy()
        y[y>2**15] = y[y>2**15] - 2**16                                         # correct for the fact that the measurement is on 16 bit

        thresh = 750                                                            # TODO: thresh is bad

        for i in range(len(self.proxv)):
            #sensor_data[i] = 1000;
            xrel = util.rot_mat2D(self.xhat[2][0]) @ self.proxv[i].pos          # estimated relative position of sensor 0
            x    = self.xhat[0:2] + xrel                                           # estimated absolute position of vertical sensor 0
            gridlines = field.grid * np.rint(x / field.grid)                    # closest gridlines
            dist = np.abs(x - gridlines)                                      # estimated distances to closest gridlines
            grid_expected = dist < field.linew/2                                # do we expect a gridline here ?

            if sensor_data[7+i] < thresh :
                #c0 = np.zeros((1,5))
                #idx = np.argmin(dist)                                           # find closes gridlie to sensor position
                #c0[0, idx] = 1;                                                 # prepare C matrix
                #y0 = np.array([gridlines[idx] - xrel[idx]])
                #if np.any(prev_sensor_data[7:9] < thresh) or grid_expected[idx]:# we can consider we are in the middle of a line
                #    r0 = np.array([(field.linew/2)**2])                         # with high certainty
                #else :
                #    r0 = np.array([(2*field.linew)**2])                         # otherwise we're not very certain (arbitrary large variance)
                c0    = np.concatenate((np.eye(2), np.zeros((2,3))), axis = 1)
                line  = np.array([np.linspace(-field.grid/2, field.grid/2, field.grid)])
                xgrid = np.concatenate((gridlines[0]*np.ones((1,field.grid)), x[1] + line), axis=0);
                ygrid = np.concatenate((x[0] + line, gridlines[1]*np.ones((1,field.grid))), axis=0);
                pdf   = np.concatenate((util.mvnpdf2D(xgrid,x,self.s[0:2,0:2]),util.mvnpdf2D(ygrid,x,self.s[0:2,0:2])))


                idx_pos = np.argmax(pdf, axis=1)
                idx_dim = np.argmax(np.array([pdf[0,idx_pos[0]], pdf[1,idx_pos[1]]]))

                #print(idx_dim)

                if idx_dim == 0:
                    y0 = np.array([xgrid[:,idx_pos[0]]]).T - xrel
                    r0 = np.array([field.linew, field.grid/4])**2
                else :
                    y0 = np.array([xgrid[:,idx_pos[1]]]).T - xrel
                    r0 = np.array([field.grid/4, field.linew])**2

                r = np.concatenate((r0, r), axis=0)                             # BEWARE : this part is not robust at all, it can go terribly wrong terribly fast
                C = np.concatenate((c0, C))
                y = np.concatenate((y0, y), axis=0)
            elif np.any(grid_expected) :                                        # if we expect a grid to be there, but it isn't
                idx = grid_expected.reshape(2)
                c0  = np.eye(2,5)[idx,:]
                y0  = (gridlines - xrel - np.sign((gridlines - x))*field.linew)[idx]
                r0  = (field.grid/4)**2 * np.ones((1,2))[:,idx][0]
                r = np.concatenate((r0, r), axis=0)
                C = np.concatenate((c0, C))
                y = np.concatenate((y0, y), axis=0)

        self.C = C
        self.R = np.diag(r)
        return y


    ###### Do real stuff with Thymio
    def do_motion(self, u):
        v = u.copy()
        v[u < 0] = 2**16 + u[u < 0]                                             # conform motor input to standard

        self.th.set_var("motor.right.target", np.uint16(v[0][0]))
        self.th.set_var("motor.left.target",  np.uint16(v[1][0]))

    def do_measure(self):

        sensor_data  = np.zeros((7+2+2,1))                                      # [0-6: proxh, 7-8: proxv, 9-10: motor_speed]

        sensor_data[0:7]    = np.array([self.th["prox.horizontal"]]).T
        sensor_data[7:9]    = np.array([self.th["prox.ground.delta"]]).T
        sensor_data[9][0]   = self.th["motor.right.speed"]
        sensor_data[10][0]  = self.th["motor.left.speed"]

        return sensor_data


    ###### Simulation stuff
    def simulate_motion(self, u=np.zeros((2,1)), q=np.array([])):
        """ simulate the motion of a robot """

        if not q.size:
            q = self.q

        w = np.sqrt([q]).T * np.random.randn(2,1)                               # simulate process noise
        self.x = self.motion_model(self.x, u + w)

        return self.x.copy()

    def simulate_measure(self, field, r=np.array([])):
        """ simulate the measurements taken by a robot in the field """

        if not r.size:
            r = self.r
        sensor_data  = np.zeros((7+2+2,1))                                      # [0-6: proxh, 7-8: proxv, 9-10: motor_speed]

        for i in range(len(self.proxh)) :                                       # measure the horizontal proximity sensors w/ noise
            sensor_data[i][0] = 950 + 20*np.random.randn()                      # return a "white" value

        for i in range(len(self.proxv)) :                                       # measure the vertical proximity sensors w/ noise
            xrel = util.rot_mat2D(self.x[2][0]) @ self.proxv[i].pos             # relative position of vertical sensor
            x    = self.x[0:2] + xrel                                           # absolute position of vertical sensor
            dist = np.abs(x - field.grid*np.rint(x/field.grid))                 # distance from gridlines
            if (dist[0] < field.linew/2) or (dist[1] < field.linew/2) :         # if we're on a gridline
                sensor_data[i+7][0] = 500 + 70*np.random.randn()                # return a "black" value
            else :
                sensor_data[i+7][0] = 950 + 20*np.random.randn()                # return a "white" value

        sensor_data[9:11] = self.x[3:5] + np.sqrt([r]).T * np.random.randn(2,1) # measure the motor speed w/ noise

        return sensor_data

    ###### System control stuff
    def kalman_estimator(self, u, y):
        """ generic extendend Kalman state estimator for a robot """

        xbar      = self.motion_model(self.xhat, u)                             # a priori estimation
        G         = self.motion_model_jacobian(self.xhat, u)
        sbar      = G @ self.s @ G.T + self.motion_model_covariance(G)

        S         = self.C @ sbar @ self.C.T + self.R
        K         = sbar @ self.C.T @ np.linalg.inv(S)                          # Kalman gain
        self.xhat = xbar + K @ (y - self.C @ xbar)                              # a posteriori estimation
        self.s    = (np.eye(5) - K @ self.C) @ sbar

        return self.xhat.copy(), self.s.copy()

    def pilot(self, goals):
        """ simplistic controller, wich takes a trajectory as input and outputs
            the motor control signals to achieve the goal, without bumping into
            stuff """

        thresh  = 50
        arrived = False

        dist = [];
        for goal in goals :                                                     # find closes active goal (in case current goal is blocked by an obstacle)
            dist.append(np.linalg.norm(np.array([goal]).T - self.xhat[0:2]))
        idx = np.argmin(np.array(dist));

        while idx < len(dist) and dist[idx] < thresh:                           # if we're close enough, we're close enough
            arrived = True
            idx = idx + 1
            #print("arrived")

        for i in range(idx) :
            goals.pop(0)                                                        # pop the useless goals from the list

        if goals :
            u = self.motion_model_inverse(self.xhat[0:3], np.array([goals[0]]).T) # use inverse motion model to figure out control signal to reach the next goal
        else:
            u = np.zeros((2,1))

        #u = self.avoid_obstacle(u, sensor_data)
        # make sure we don't bump into anything unexpected

        # check if thymio is avoiding an obstacle
        #  if yes and the goal is near
        #    -> check if goal is on the left or on the right
        #       -> exit obstacle avoidance or not

        local_obstacle = False

        if self.th["event.args"][0] == 1:
            local_obstacle = True
            if arrived:
                side = self.th["event.args"][1] # 0 when left, 1 when right, 3 when none
                goal_side = side

                delta_x = (self.xhat[0] - np.array([goals[0]]).T[0])
                delta_y = (self.xhat[1] - np.array([goals[0]]).T[1])
                robot_ang = self.xhat[2]
                ang_to_goal = math.atan2(delta_y, delta_x) # between -pi and pi
                while ang_to_goal > 2*math.pi:
                    ang_to_goal = ang_to_goal - 2*math.pi
                while ang_to_goal < 0:
                    ang_to_goal = ang_to_goal + 2*math.pi
                while robot_ang > 2*math.pi:
                    robot_ang = robot_ang - 2*math.pi
                while robot_ang < 0:
                    robot_ang = robot_ang + 2*math.pi

                delta_ang = ang_to_goal - robot_ang
                while delta_ang > 2*math.pi:
                    delta_ang = delta_ang - 2*math.pi
                while delta_ang < 0:
                    delta_ang = delta_ang + 2*math.pi
                if delta_ang > math.pi:
                    goal_side = 1 # goal on the right
                if delta_ang < math.pi:
                    goal_side = 0 # goal on the left

                if side != goal_side:
                    self.th.set_var("event.args", 2)
                    local_obstacle = False

        return u, local_obstacle
