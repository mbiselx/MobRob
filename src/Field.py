import matplotlib.pylab as plt
import numpy as np

class NewField:
    def __init__ (self, xsize, ysize, grid, linew):
        """ make a simplistic model of the robot's environment """
        self.xmin  = 0;                                                         # field size
        self.xmax  = xsize;
        self.ymin  = 0;
        self.ymax  = ysize;

        self.bkgnd = np.array([])                                               # field elements
        self.obst  = np.array([])
        self.goals = []
        self.grid  = grid
        self.linew = linew

        self.xreal = np.array([])                                               # robot stuff
        self.xhat  = np.array([])
        self.xodo  = np.array([])
        self.s     = np.array([])


    def plot(self):
        s = np.array(self.s)
        xreal = self.xreal.copy()
        xodo  = self.xodo.copy()
        xhat  = self.xhat.copy()
        goals = np.array(self.goals).T
        bkgnd = self.bkgnd.copy()

        #if bkgnd.size :
        #    plt.imshow(bkgnd, origin = 'lower', extent=(field.xmin, field.xmax, field.ymin, field.ymax))

        for i in range(int((self.xmax - self.xmin)/self.grid)+1) :                   # plot vertical gridlines
            plt.plot(i*self.grid*np.ones((2,1)), np.array([self.ymin, self.ymax]), '-k');
        for i in range(int((self.ymax - self.ymin)/self.grid)+1) :                   # plot horizontal gridlines
            plt.plot(np.array([self.xmin, self.xmax]), i*self.grid*np.ones((2,1)), '-k');

        if s.size and s.size == xhat.shape[1]:
            cos, sin = np.cos(xhat[2,:]), np.sin(xhat[2,:])
            plt.plot(xhat[0,:] - s*sin, xhat[1,:] + s*cos, '--c', label="uncertainty"); # plot uncertainty
            plt.plot(xhat[0,:] + s*sin, xhat[1,:] - s*cos, '--c');

        if xreal.size :
            plt.plot(xreal[0,:], xreal[1,:], '-g', linewidth=2, label="camera trajectory")          # plot real trajectory

        if xodo.size :
            plt.plot(xodo[0,:],  xodo[1,:],  '-y', linewidth=2, label="odometry trajectory");

        if xhat.size :
            plt.plot(xhat[0,:],  xhat[1,:],  '-r', label="estimated trajectory");                 # plot estimated trajectory

        if goals.size :
            plt.scatter(goals[0,:], goals[1,:], marker="o", color='b', label="goals");                          # plot goals

        plt.legend(loc="upper right");
        plt.axis("equal");
