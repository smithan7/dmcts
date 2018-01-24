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
roslaunch hector_quadrotor_gazebo mySpawn_quadrotor_with_kinect.launch &
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

echo "launching rviz"
rviz &
pid="$pid $!"

sleep 5s

echo "launching pid controller"
python ~/catkin_ws/src/hector_quadrotor/hector_quadrotor_gazebo/launch/pid_controller.py &
pid="$pid $!"

trap "echo Killing all processes.; kill -2 TERM $pid; exit" SIGINT SIGTERM

sleep 24h