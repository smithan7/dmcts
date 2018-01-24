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

	int test_environment_number = 0;
	int agent_index = 1;
	int params = 0;
	bool display_map = true;
	bool score_run = false;
	int n_nodes = 1;
	int n_agents = 0;

	//std::string task_selection_method = "greedy_completion_reward";
	std::string task_selection_method = "mcts_task_by_completion_reward_impact_optimal";
	std::string world_directory, map_name;

	ros::param::get("test_environment_number", test_environment_number);
	ros::param::get("agent_index", agent_index);
	ros::param::get("world_directory", world_directory);
	ros::param::get("score_run", score_run);
	ros::param::get("display_map", display_map);
	int parameter_seed = 0;
	ros::param::get("/dmcts/parameter_seed", parameter_seed);
	ros::param::get("number_of_nodes", n_nodes);
	ros::param::get("number_of_agents", n_agents);


	ROS_INFO("World::initializing agent's world");
	ROS_INFO("   test_environment_number %i", test_environment_number);
	ROS_INFO("   agent_index %i", agent_index);
	ROS_INFO("   world directory %s", world_directory.c_str());
	ROS_INFO("   score_run %i", score_run);
	ROS_INFO("   display_map %i", display_map);
	ROS_INFO("   parameter_seed %i", parameter_seed);
	ROS_INFO("   n_nodes %i", n_nodes);
	ROS_INFO("   n_agents %i", n_agents);

	World world = World(nHandle, parameter_seed, display_map, score_run, task_selection_method, world_directory, agent_index, n_nodes, n_agents);

	// return the control to ROS
	ros::spin();

	return 0;
}
