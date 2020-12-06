import cv2
import numpy as np

import Vision as vis

# get img from the webcam
cap_size  = (1200, 1000)
map_shape = [1200,  800]
cap       = cv2.VideoCapture(2)                       # put 1 if an external webcam is used
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_size[0])        # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cap_size[1])        # height
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)                 # set auto exposure

robot_pos = np.array([])

while True :
    success, img = cap.read()
    if success :
        success2, true_map = vis.get_clean_map(img, map_shape)                  # get a clean map
        if success2 :
            robot_pos = vis.robot_detection(true_map)                           # find the robot, annotate on map
            goal_pos  = vis.goal_detection(true_map)                            # find the goal, annotate on map
            if robot_pos.size :
                obs_map = vis.get_global_search_map(true_map, robot_pos)        # find obstacles
                cv2.imshow("Obstacles", obs_map)

            true_map = vis.annotate_map(true_map, robot_pos, goal_pos)


        cv2.imshow("Map", true_map)
        cv2.imshow("Raw", img)                                    # display

    else:
        print("Couldn't capture image")
        break

    if (cv2.waitKey(1) & 0xFF) == ord('q'):                       # quit when 'Q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
