import numpy as np

_EPSIL = 1e-6


def mvnpdf2D(x, mu, sigma):
    """ calculate multivariate normal probability """
    result = []
    x_mu = x - mu
    c = 1/(2*np.pi*np.prod(np.diag(sigma)))
    for i in range(x.shape[1]):
        result.append(c * np.exp(-.5 * x_mu[:,i].T@ np.linalg.inv(sigma) @ x_mu[:,i]))
    return np.array([result])

def rot_mat2D(angle):
    """ make a 2D rotation matrix """
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s],
                  [s,  c]])
    return R

def avoid_singularity(variable):
    """ make a variable nonzero """
    if np.abs(variable) < _EPSIL:
        return _EPSIL
    return variable

def cap_abs(variable, cap):
    """ cap a variable to be in [-cap:cap] """
    if  variable >  cap :
        variable =  cap
    if  variable < -cap :
        variable = -cap
    return variable

def limit_pi(angle):
    """ limit angle to [-pi:pi] """
    angle = (angle + np.pi) % (2*np.pi) - np.pi
    return angle

def adapt_vision_coords(coords, field):
    """ switch from vision coordinate system to controller coordinate system """
    x    = np.zeros((5,1))
    x[0] = coords[0]
    x[1] = coords[1]
    x[2] = limit_pi(coords[2])
    return x

def adapt_pathfinder_coords(path, field, grid_shape):
    """ switch from pathfinder coordinate system to controller coordinate system """
    path      = np.fliplr(np.flipud(path.T))                                    # switch left-right & up-down
    path[:,0] = path[:,0]*field.xmax/grid_shape[1]                              # convert path from pixel of grid (max_val_row x max_val_column) into pixel of field (800mm x 1200mm)
    path[:,1] = path[:,1]*field.ymax/grid_shape[0]
    return path

def rescale_int_coords(coords, map_size_in, map_size_out):
    """ rescale coordinates  """
    coords[0] = coords[0] * map_size_out[0] / map_size_in[0]
    coords[1] = coords[1] * map_size_out[1] / map_size_in[1]
    return np.int32(coords)
