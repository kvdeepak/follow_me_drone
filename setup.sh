mkdir -p ./bebop_2_follow_me/src && cd ./bebop_2_follow_me
catkin init
git clone https://github.com/AutonomyLab/bebop_autonomy.git src/bebop_autonomy
git clone https://github.com/ros-teleop/teleop_twist_keyboard src/teleop_twist_keyboard
cd src/
catkin_create_pkg image_subscriber sensor_msgs cv_bridge rospy std_msgs
cd image_subscriber/
wget download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz
mv ../../../human-detection.py .
cd ../teleop_twist_keyboard/
git apply ../../../teleop.diff
rosdep update
rosdep install --from-paths src -i
catkin build
mv control_quad.sh init_driver.sh simulate_quad.sh start_firmwared.sh bebop_2_follow_me/
