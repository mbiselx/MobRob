# Mobile Robotics Project
## for the EPFL course Basics of Mobile Robotics [MICRO-452](https://isa.epfl.ch/imoniteur_ISAP/!itffichecours.htm?ww_i_matiere=2343869959&ww_x_anneeAcad=2020-2021&ww_i_section=2373753419)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project
The objective of this project is to create a program in Python/Aseba that:
* Uses the visual feedback of a webcam to detect the starting position of a robot, its goal position and 2D obstacles in a given environment with some known specifications described in the following section Environment.
* Finds the optimal path from the start position of the robot to its goal between the global obstacles. It uses an A-star algorithm.
* Controls the robot to follow the optimal path, by successively applying filters to estimate its position. The entries are the absolute position of the robot given by the camera, the odometry and proximity and black and white sensors underneath the robot.
* Navigates around 3D local obstacles using local avoidance.


<!-- FOLDER Structure -->
## Folder Structure
| Folder Name             | Comment                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Aseba                   | Aseba code for the local obstacle avoidance                                                        |
| imgs                    | Images used in the Jupyter Notebook                                                                |
| src                     | Python files used in the Jupyter Notebook                                                          |
| videos                  | Videos used in the Jupyter Notebook                                                                |


<!-- CONTACT -->
## Contact
Biselx Michael - michael.biselx@epfl.ch <br />
Samuel Bumann - samuel.bumann@epfl.ch




Project Link: [https://github.com/mbiselx/MobRob](https://github.com/mbiselx/MobRob)
