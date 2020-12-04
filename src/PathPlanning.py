import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors


################################################################################
##  functions for path planning
################################################################################

def generate_global_path(start, goal, occupancy_grid):
    #get max_val_row and column
    max_val_row=occupancy_grid.shape[0]
    max_val_column=occupancy_grid.shape[1]

    # EXECUTION AND PLOTTING OF THE ALGORITHM
    # List of all coordinates in the grid
    x,y=np.mgrid[0:max_val_column:1, 0:max_val_row:1]
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = y
    pos[:,:,1] = x
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])


    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run the A* algorithm
    path, visitedNodes = A_Star(start, goal, h, coords, occupancy_grid, max_val_column, max_val_row, movement_type="8N")
    path = np.array(path).reshape(-1, 2).transpose()
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()

    # Displaying the map
    fig_astar, ax_astar = create_empty_plot(max_val_row, max_val_column)
    ax_astar.imshow(occupancy_grid, cmap=colors.ListedColormap(['white', 'red']))

    # Plot the best path found and the list of visited nodes
    ax_astar.scatter(visitedNodes[1], visitedNodes[0], marker="o", color = 'orange');
    ax_astar.plot(path[1], path[0], marker="o", color = 'blue');
    ax_astar.scatter(start[1], start[0], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[1], goal[0], marker="o", color = 'purple', s=200);

    return path

def A_Star(start, goal, h, coords, occupancy_grid, max_val_column, max_val_row, movement_type="8N"):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node, tuple (x, y)
    :param goal_m: goal node, tuple (x, y)
    :param occupancy_grid: numpy array containing the map with the obstacles. At each position,
                           you either have the number 1 (occupied) or 0 (free)
    :param movement: string, select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """

    # Check if the start and goal are within the boundaries of the map
#    for point in [start, goal]:
#        assert point[0]>=0 and point[0]<max_val_column, "start or end goal not contained in the map"
#        assert point[1]>=0 and point[1]<max_val_row, "start or end goal not contained in the map"

    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        print(start[0], start[1])
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        print(goal[0], goal[1])
        raise Exception('Goal node is not traversable')

    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')


    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------


    # Here we initialise the variables, feel free to print them or use the variable info function if it is not
    # what they contain or how to access the different elements

    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]

    # The set of visited nodes that no longer need to be expanded.
    # Note that this is an addition w.r.t. the wikipedia pseudocode
    # It contains the list of variables that have already been visited
    # in order to visualise them in the plots
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]


    # while there are still nodes to visit
    while len(openSet)!=0:

        #find the unvisited node having the lowest fScore[] value
        node_min_fScore_value=np.inf
        for node in openSet:
            if fScore[node]<node_min_fScore_value:
                node_min_fScore_value=fScore[node]
                current=node

        #If the goal is reached, reconstruct and return the obtained path
        if current==goal:
            path=[goal]
            while current!=start:
                current=cameFrom[current]
                path.append(current)
            return path, closedSet
        openSet.remove(current)
        closedSet.append(current)
        # If the goal was not reached, for each neighbor of current:
        for dx, dy, deltacost in movements:

            neighbor = (current[0]+dx, current[1]+dy)

            # if the node is not in the map, skip
            if neighbor not in coords:
                continue

            # if the node is occupied or has already been visited, skip
            if occupancy_grid[neighbor[0], neighbor[1]] or neighbor in closedSet:
                continue

            # compute the cost to reach the node through the given path
            tentative_gScore = gScore[current]+deltacost

            # If the computed cost is the best one for that node, then update the costs and
            # node from which it came
            if tentative_gScore<gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor]=current
                gScore[neighbor]=tentative_gScore
                fScore[neighbor]=gScore[neighbor]+h[neighbor]
                if neighbor not in openSet:
                    openSet.append(neighbor)

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], []

def create_empty_plot(max_val_row, max_val_column):
    """
    Helper function to create a figure of the desired dimensions & grid

    :param max_val_row: dimension of the map along the y dimension
    :param max_val_column: dimension of the map along the x dimension
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(15,10))

    major_ticks_row = np.arange(0, max_val_row+1, 5)
    minor_ticks_row = np.arange(0, max_val_row+1, 1)
    major_ticks_column = np.arange(0, max_val_column+1, 5)
    minor_ticks_column = np.arange(0, max_val_column+1, 1)
    ax.set_xticks(major_ticks_column)
    ax.set_xticks(minor_ticks_column, minor=True)
    ax.set_yticks(major_ticks_row)
    ax.set_yticks(minor_ticks_row, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([0,max_val_row])
    ax.set_xlim([0,max_val_column])
    ax.grid(True)

    return fig, ax

def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]

def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


################################################################################
##  functions for testing
################################################################################

def idx_neighborhood(pos, size):
    """
    generates the indexes of the neighborhood around a given position
    :param pos: position around which the indxses of the neighborhood will be generated
    :param size: size of the neighborhood
    :return: array with indexes of elements of neighborhood
    """
    idx=[[] for _ in range(size)]
    for column in range(size):
        for row in range(size):
            idx[column].append((pos[0]-math.floor(size/2)+row, pos[1]+math.floor(size/2)-column))

    return idx

def incert_array(array, array_to_incert, pos):
    """
    makes a logic OR of the array_to_invert and the neighborhood around pos of array
    :param array: array which is modified
    :param array_to_incert: array that is incerted
    :param pos: position where array_to_incert is inserted
    :return: array with array_to_incert incerted (at position pos)
    """
    idx=idx_neighborhood(pos, len(array_to_incert))
    for column, val_row in enumerate(idx):
        for row, pos_idx in enumerate(val_row):
            if pos_idx[0]>=array.shape[0] or pos_idx[1]>=array.shape[1]:
                continue
            if pos_idx[0]<0 or pos_idx[1]<0:
                continue
            array[pos_idx]+=array_to_incert[(row,len(idx)-1-column)]
    array[array>=1]=1
    return array

def creat_random_object(size):#creates random object
    """
    generates a random object saved in array of given size
    :param size: arrays proportion
    :return: array containing random object
    """
    limit=90
    if size==0:
        return np.zeros((1, 1))
    ran_object=np.zeros((size, size))
    ran_object[round(size/2), round(size/2)]=1#"center" of object
    for k in range(round(size/2)):
        for column, val_row in enumerate(ran_object):
            for row, val_pixel in enumerate(val_row):
                if val_pixel==1:
                    data = np.random.rand(3, 3)*100
                    data[data<limit]=0
                    data[data>=limit]=1
                    ran_object=incert_array(ran_object, data, (row, column))
    return ran_object

def create_random_occupancy_grid(max_val_column, max_val_row):
    """
    generates a occupancy_grid with a random amound of objects of random size and position
    :param max_val_column: maximum number of columns/ maximal value of x
    :param max_val_row: maximum number of rows/ maximal value of y
    :return occupancy_grid: array containing map
    :return cmap: color map of objects and free spaces
    """
    fig, ax = create_empty_plot(max_val_row, max_val_column)

    # Creating the occupancy grid
    # np.random.seed(0) # To guarantee the same outcome on all computers
    data = np.random.rand(max_val_row, max_val_column) * 1000 # Create a grid of 50 x 50 random values
    cmap = colors.ListedColormap(['white', 'red']) # Select the colors with which to display obstacles and free cells

    # Converting the random values into occupied "centers" and free cells
    limit = 999
    occupancy_grid = data.copy()
    occupancy_grid[data>limit] = 1
    occupancy_grid[data<=limit] = 0
    occupancy_grid_temp = occupancy_grid.copy()

    for column, val_row in enumerate(occupancy_grid):
        for row, val_pixel in enumerate(val_row):
            if val_pixel==1:
                occupancy_grid_temp=incert_array(occupancy_grid_temp, creat_random_object(np.random.randint(max_val_row/6,max_val_row/3)), (column, row))
    occupancy_grid=occupancy_grid_temp
    # Displaying the map
    ax.imshow(occupancy_grid, cmap=cmap)
    plt.title("Map : free cells in white, occupied cells in red");

    return occupancy_grid, cmap
