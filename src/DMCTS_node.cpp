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

	int test_environment_number, agent_index, params, n_nodes,n_agents, parameter_seed;
	bool display_map, score_run, pay_obstacle_costs;
	std::string task_selection_method;
	double desired_alt, p_task_initially_active, cruising_speed;

	//std::string task_selection_method = "greedy_completion_reward";
	//std::string task_selection_method = "mcts_task_by_completion_reward_impact_optimal";
	std::string world_directory, map_name;

	ros::param::get("test_environment_number", test_environment_number);
	ros::param::get("agent_index", agent_index);
	ros::param::get("world_directory", world_directory);
	ros::param::get("score_run", score_run);
	ros::param::get("display_map", display_map);
	ros::param::get("param_number", parameter_seed);
	ros::param::get("number_of_nodes", n_nodes);
	ros::param::get("number_of_agents", n_agents);
	ros::param::get("coord_method", task_selection_method);
	ros::param::get("desired_altitude", desired_alt);
	ros::param::get("p_task_initially_active", p_task_initially_active);
	ros::param::get("pay_obstacle_costs", pay_obstacle_costs);
	ros::param::get("cruising_speed", cruising_speed);

	ROS_INFO("World::initializing agent's world");
	ROS_INFO("   test_environment_number %i", test_environment_number);
	ROS_INFO("   agent_index %i", agent_index);
	ROS_INFO("   world directory %s", world_directory.c_str());
	ROS_INFO("   score_run %i", score_run);
	ROS_INFO("   display_map %i", display_map);
	ROS_INFO("   parameter_seed %i", parameter_seed);
	ROS_INFO("   n_nodes %i", n_nodes);
	ROS_INFO("   n_agents %i", n_agents);
	ROS_INFO("   coord_method %s", task_selection_method.c_str());
	ROS_INFO("   desired_altitude %.1f", desired_alt);
	ROS_INFO("   p_task_initially_active %0.4f", p_task_initially_active);
	ROS_INFO("   pay_obstacle_costs %i", pay_obstacle_costs);
	ROS_INFO("   cruising_speed %0.1f", cruising_speed);

	World world = World(nHandle, parameter_seed, display_map, score_run, task_selection_method, world_directory, agent_index, n_nodes, n_agents, desired_alt, p_task_initially_active, pay_obstacle_costs, cruising_speed);

	// return the control to ROS
	ros::spin();

	return 0;
}
