#!/bin/bash
 
my_pid=$$
echo "My process ID is $my_pid"

echo "Sourcing ~/catkin_ws/devel/setup.bash"
source ~/catkin_ws/devel/setup.bash &
pid=$!

echo "Launching roscore..."
roscore &
pid="pid $!"


sleep 5s

echo "launching hector quadrotor with kinect"
roslaunch hector_quadrotor_gazebo spawn_quadrotor_with_asus.launch &
pid="$pid $!"

sleep 5s

echo "Launching Gazebo..."
roslaunch gazebo_ros willowgarage_world.launch &
pid="$pid $!"

sleep 5s

echo "launching rtabmap"
roslaunch hector_quadrotor_gazebo rtabmap_ground_truth_andy.launch &
pid="$pid $!"

sleep 5s

echo "fixing movebase transform"
rosrun tf static_transform_publisher 0 0 0 0 0 0 /base_link odom 100 &
pid="$pid $!"
sleep 1s

echo "launching laser scan from depth image"
rosrun depthimage_to_laserscan depthimage_to_laserscan image:=/camera/depth/image_raw camera_info:=/camera/depth/camera_info &
roslaunch dmcts_world scan_from_depth_image.launch &
pid="$pid $!"

sleep 1s

echo "launching rviz"
rviz &
pid="$pid $!"

sleep 5s

echo "launching pid controller"
python ~/catkin_ws/src/my_quad_controller/scripts/move_base_path_through.py &
pid="$pid $!"
sleep 10s

trap "echo Killing all processes.; kill -2 TERM $pid; exit" SIGINT SIGTERM

sleep 24h
