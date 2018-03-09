#include <ros/ros.h>
#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/Pose.h"
#include <move_base_msgs/MoveBaseAction.h>
#include <costmap_2d/costmap_2d.h>

#include "World.h"
#include "Agent.h"


int main(int argc, char *argv[]){
	// initialization
	ros::init(argc, argv, "Agent");
	ros::NodeHandle nHandle("~");

	World world = World(nHandle);

	// return the control to ROS
	ros::spin();

	return 0;
}
