# Follow me using Parrot Bebop 2 Quadcopter

Instructions:
1) Make sure you have ros kinetic, sphinx(for simulation of parrot bebop 2) and opnecv preinstalled
2) Run the setup.sh file to create your catkin workspace
3) Connect to bebop 2 wifi and go into bebop_2_follow_me/
Run the each of the following scripts in a  new terminal: 
4) Run start_firmwared.sh in one terminal to start firmared process
5) Run init_driver.sh in a new terminal to init bebop 2 drivers
6) Run control_quad.sh to control the quadcopter with your keyboard
7) Run human_detect.sh to view camera output and observe bounding boxes around humans

TODO:
Agent function to control motion of quad

Important links:
1) https://bebop-autonomy.readthedocs.io/en/latest/
2) http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
3) https://github.com/ros-teleop/teleop_twist_keyboard
