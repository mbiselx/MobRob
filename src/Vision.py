import cv2
from skimage import morphology
import numpy as np


def get_clean_map(img, map_shape) :
    """ find the blue corners and distort the image to get a
    nice map. return true_map, the map in high resolution """

    blurred_img = cv2.GaussianBlur(img,(15,15),cv2.BORDER_DEFAULT)              # blur img to get more uniform values

    imgHSV = cv2.cvtColor (blurred_img, cv2.COLOR_BGR2HSV )
    lower  = np.array([ 85, 75, 75])                                            # h s v
    upper  = np.array([120,255,255])                                            # hue saturation value
    mask   = cv2.inRange(imgHSV, lower, upper)                                  # mask to get the blue color

    mask   = cv2.erode( mask, np.ones((10,10),np.uint8), iterations = 1)        # erode and dilate to get rid of small blue traces
    mask   = cv2.dilate(mask, np.ones((10,10),np.uint8), iterations = 1)

    labeled_mask   = morphology.label(mask, background=0)                       # label all contingent blobs
    labels, counts = np.unique(labeled_mask, return_counts = True)
    labels = labels[np.argsort(counts)];                                        # sort the blobs by size,

    if (labels.size < 5):
        return img.copy()

    blobs = np.zeros((4,2))
    for i in range(0,4):                                                        # take the four biggest blobs
        blob = np.where(labeled_mask == (labels[labels.size-2-i]),1,0)          # excluding the blob of blackness, which is the largest

        M = cv2.moments(np.float32(blob))                                       # compute the centroids of each blob, they are the corner of the map
        cX =  int ( M ["m10"] / M ["m00"] )
        cY =  int ( M ["m01"] / M ["m00"] )
        blobs[i , :] = np.array([cX,cY])

    xmin, xmax = np.min(blobs[:,0]), np.max(blobs[:,0])                         # sort corners
    ymin, ymax = np.min(blobs[:,1]), np.max(blobs[:,1])
    x_middle = (xmax-xmin)/2+xmin
    y_middle = (ymax-ymin)/2+ymin

    corners = np.zeros((4,2))
    for corner in blobs :                                                       # figure out which corner goes where
        if   (corner[0] < x_middle) and (corner[1] < y_middle):
            corners[0,:] = corner
        elif (corner[0] < x_middle) and (corner[1] > y_middle) :
            corners[1,:] = corner
        elif (corner[0] > x_middle) and (corner[1] < y_middle) :
            corners[2,:] = corner
        elif (corner[0] > x_middle) and (corner[1] > y_middle) :
            corners[3,:] = corner

    target_shape = np.array([[0,0],[0,map_shape[1]], [map_shape[0],0], map_shape], np.float32)
    transformer  = cv2.getPerspectiveTransform(np.float32(corners), target_shape)
    true_map     = cv2.warpPerspective(img, transformer, tuple(map_shape))      # apply reshape
    return true_map


def robot_detection(true_map, draw = False):
    """ return the position of the red robot (x,y,angle) """

    blurred_img = cv2.GaussianBlur(true_map,(15,15),cv2.BORDER_DEFAULT)         # blur img to get more uniform values

    imgHSV2 = cv2.cvtColor (blurred_img, cv2.COLOR_BGR2HSV )
    lower1  = np.array ([  0 ,  75 ,  75])                                      # hsv for lower red-orange
    upper1  = np.array ([ 20 , 255 , 255])
    lower2  = np.array ([170 ,  75 ,  75])                                      # hsv for upper red-magenta
    upper2  = np.array ([180 , 255 , 255])
    mask1   = cv2.inRange ( imgHSV2 , lower1 , upper1 )
    mask2   = cv2.inRange ( imgHSV2 , lower2 , upper2 )
    mask    = cv2.bitwise_or(mask1, mask2)                                      # mask to get the red color

    mask    = cv2.erode( mask, np.ones((10,10),np.uint8), iterations = 1)       # erode and dilate to get rid of small red traces
    mask    = cv2.dilate(mask, np.ones((10,10),np.uint8), iterations = 1)

    labeled_mask   = morphology.label(mask, background=0)                       # label all contingent blobs
    labels, counts = np.unique(labeled_mask, return_counts = True)
    labels = labels[np.argsort(counts)];                                        # sort the blobs by size,

    if (labels.size < 3):
        if draw :
            return np.array([]), true_map
        else :
            return np.array([])                                                 # return empty array

    blobs = np.zeros((2,2))
    for i in range(0,2):                                                        # take the two biggest blobs
        blob = np.where(labeled_mask == (labels[labels.size-2-i]),1,0)          # excluding the blob of blackness, which is the largest
        M = cv2.moments(np.float32(blob))                                       # compute the centroids of each blob, they are the robot identifiers
        cX =  int ( M ["m10"] / M ["m00"] )
        cY =  int ( M ["m01"] / M ["m00"] )
        blobs[i , :] = np.array([cX,cY])

    robot_pos = np.zeros((5,1))                                                 # for interfacing with the rest of the project, two last values are 0, as they represent the speed of the robot
    robot_pos[0] = blobs[0,0] - 20/400*(blobs[0,0] - 600)                       # the biggest blob give the position (x,y) of the robot
    robot_pos[1] = blobs[0,1] + 35/600*(800 - blobs[0,1])                       # the non-orthogonality of the camera needs to be corrected (these are heuristic values)
    dx = blobs[1, 0:2] - blobs[0, 0:2]                                          # the smaller blob can be used to calculate the angle
    robot_pos[2] = -np.arctan2(dx[1], dx[0])

    if draw :
        r_top =  (int(robot_pos[0][0] - 10), int(robot_pos[1][0] - 10))
        r_txt =  (int(robot_pos[0][0] - 10), int(robot_pos[1][0] - 20))
        r_bot =  (int(robot_pos[0][0] + 10), int(robot_pos[1][0] + 10))
        true_map = cv2.rectangle(true_map, r_top, r_bot, (255,0,0), 2)
        cv2.putText(true_map, "robot", r_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2);
        robot_pos[1] = true_map.shape[0] - robot_pos[1]
        return robot_pos, true_map
    else :
        robot_pos[1] = true_map.shape[0] - robot_pos[1]
        return robot_pos


def goal_detection(true_map, draw = False):
    """ return the position of the green goal (x,y) """

    blurred_img = cv2.GaussianBlur(true_map,(15,15),cv2.BORDER_DEFAULT)         # blur img to get more uniform values

    imgHSV = cv2.cvtColor (blurred_img, cv2.COLOR_BGR2HSV )
    lower  = np.array([ 40, 50, 50])                                            # h s v
    upper  = np.array([ 80,255,255])                                            # hue saturation value
    mask   = cv2.inRange(imgHSV, lower, upper)                                  # mask to get the blue color

    mask   = cv2.erode( mask, np.ones((10,10),np.uint8), iterations = 1)        # erode and dilate to get rid of small green traces
    mask   = cv2.dilate(mask, np.ones((10,10),np.uint8), iterations = 1)

    labeled_mask   = morphology.label(mask, background=0)                       # label all contingent blobs
    labels, counts = np.unique(labeled_mask, return_counts = True)
    labels = labels[np.argsort(counts)];                                        # sort the blobs by size,

    if (labels.size < 2):
        if draw :
            return np.array([]), true_map
        else :
            return np.array([])                                                 # return empty array

    blob = np.where(labeled_mask == (labels[labels.size-2]),1,0)                # take second-largest blob (largest is background)
    M = cv2.moments(np.float32(blob))                                           # compute the centroid of the blob
    cX =  int ( M ["m10"] / M ["m00"] )
    cY =  int ( M ["m01"] / M ["m00"] )

    goal = np.array([[cX] , [cY]])

    if draw :
        g_top =  (int(goal[0][0] - 10), int(goal[1][0] - 10))
        g_txt =  (int(goal[0][0] - 10), int(goal[1][0] - 20))
        g_bot =  (int(goal[0][0] + 10), int(goal[1][0] + 10))
        true_map = cv2.rectangle(true_map, g_top, g_bot, (255,0,0), 2)
        cv2.putText(true_map, "goal", g_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2);
        goal[1] = true_map.shape[0] - goal[1]
        return goal, true_map
    else :
        goal[1] = true_map.shape[0] - goal[1]
        return goal


def get_global_search_map(true_map):
    """ return the map dilate to make obstacles a bit bigger to take into
    account the size of the robot """

    processed_map = cv2.cvtColor (true_map, cv2.COLOR_BGR2GRAY )                # make img grayscale
    processed_map = cv2.GaussianBlur(processed_map,(15,15),cv2.BORDER_DEFAULT)  # blur img to get more uniform values

    _ , processed_map = cv2.threshold(processed_map, 75, 255, cv2.THRESH_BINARY_INV ) # apply a fixed threshold binarization

    processed_map = cv2.erode (processed_map, np.ones((10, 10), np.uint8), iterations = 1 ) # get ride of all the thin black lines & shadows
    processed_map = cv2.dilate(processed_map, np.ones((20, 20), np.uint8), iterations = 2 ) # increase the size of the remaining obstacles

    return processed_map
