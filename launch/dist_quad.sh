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

echo "launching hector quadrotor"
roslaunch hector_quadrotor_gazebo spawn_quadrotor.launch &
pid="$pid $!"

sleep 5s

echo "Launching Gazebo..."
roslaunch gazebo_ros empty_world.launch &
pid="$pid $!"

sleep 5s

echo "launching rviz"
rviz &
pid="$pid $!"

sleep 5s

echo "launching DIST-MCTS world node"
python ~/catkin_ws/src/dmcts_world/src/master_node.py &
pid="$pid $!"

sleep 5s

echo "launching pid controller"
python ~/catkin_ws/src/hector_quadrotor/hector_quadrotor_gazebo/launch/pid_controller.py &
pid="$pid $!"

trap "echo Killing all processes.; kill -2 TERM $pid; exit" SIGINT SIGTERM

sleep 24h
