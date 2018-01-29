#pragma once



// ros stuff
#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "custom_messages/DMCTS_Probability.h"
#include "custom_messages/DMCTS_Pulse.h"
#include "rosgraph_msgs/Clock.h"

#include <vector>
// opencv stuff
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>



class World;
class Agent_Planning;
class Agent_Coordinator;
class Goal;
class Pose;

class Agent
{
public:
	// ros stuff
	ros::Timer plan_timer, act_timer, send_loc_timer, task_list_timer;
	ros::Publisher move_pub, coord_pub;
	ros::Subscriber odom_sub, coord_sub, pulse_sub; //clock_sub;
	ros::ServiceClient task_list_client, work_client, send_loc_client, a_star_path_client, kinematic_client;
	ros::Duration act_duration, plan_duration, send_loc_duration, task_list_duration;
	void plan_timer_callback(const ros::TimerEvent &e); // use planning method
	void act_timer_callback(const ros::TimerEvent &e); // move / work
	void publish_to_control_script(const int &goal_node ); // publish command vel to python node
	void publish_work_request(const int &goal_node ); // tell master node where I am and try to work
	void publish_coord_plan();
	void coord_plan_callback(const custom_messages::DMCTS_Probability &msg);
	void odom_callback(const nav_msgs::Odometry &locIn); // get my current location from gazebo
	void pulse_callback(const custom_messages::DMCTS_Pulse &msg);
	//void clock_callback(const rosgraph_msgs::Clock &tmIn);
	void send_request_for_task_list(); // get task list from master node, has all active tasks
	void send_loc_service_timer_callback(const ros::TimerEvent &e); // send the location to the master node
	void task_list_timer_callback(const ros::TimerEvent &e);
	void upload_new_plan(const std::vector<int> &claimed_tasks, const std::vector<double> &claimed_time, const std::vector<double> &claimed_probability);

	double map_offset_x, map_offset_y;

	// functions
	Agent(ros::NodeHandle nHandle, const int &index, const int &type, const double &travel_vel, const cv::Scalar &color, const bool &pay_obstacle_cost, const double &work_radius, const bool &actual_agent, World* world_in);
	bool at_node(int node);
	bool act();
	bool plan();
	~Agent();

	// access private variables
	Pose* get_pose() { return this->pose; };
	void set_pose(Pose* pose_in) {this->pose = pose_in; };
	void update_pose(const double &xi, const double &yi, const double &zi, const double wi);

	bool task_list_initialized, act_initialized, plan_initialized, m_node_initialized, initialized, location_initialized;

	int get_index() { return this->index; };
	cv::Point2d get_loc2d();
	int get_loc() { return this->edge.x; };

	Goal* get_goal() { return this->goal_node; };
	Agent_Coordinator* get_coordinator() { return this->coordinator; };
	Agent_Planning* get_planner() { return this->planner; };
	World* get_world() { return this->world; };
	double get_arrival_time() { return this->arrival_time; }
	
	int get_type() { return this->type; };
	double get_travel_vel() { return this->travel_vel; };
	double get_travel_step() { return this->travel_step; };
	bool get_pay_obstacle_cost() { return this->pay_obstacle_cost; };

	bool get_at_node() { return this->at_node(); };
	cv::Scalar get_color() { return this->color; };
	cv::Point2i get_edge() { return this->edge; };
	std::string get_task_selection_method() { return this->task_selection_method; };
	std::string get_task_claim_method() { return this->task_claim_method; };
	std::string get_task_claim_time() { return this->task_claim_time; };

	double get_work_done() { return this->work_done; };
	double get_travel_done() { return this->travel_done; };
	double get_collected_reward() { return this->collected_reward;};
	void set_collected_reward(const double &col_rew_in) { this->collected_reward += col_rew_in; };
	
	double get_edge_progress() { return this->edge_progress; };
	int get_edge_x() { return this->edge.x; };
	int get_edge_y() { return this->edge.y; };
private:
	// my location
	Pose* pose;

	// planning and coordinator
	Goal* goal_node;
	Agent_Planning* planner;
	Agent_Coordinator* coordinator;
	World* world;

	std::string task_selection_method; // how do I select tasks
	std::string task_claim_method; // how / when do I claim tasks
	std::string task_claim_time; // when do I claim my task
	
	double arrival_time; // for travel and arrival plan
	double expected_value;
	double work_radius; // how close can I be and still do work?

	double work_done; // accumulated reward
	double travel_done; // distance I have travelled
	double collected_reward; // how much reward I have collected

	cv::Point2i edge; // x:=where am I? y:=where am I going?
	double edge_progress; // how far along the edge am I?
	
	int index; // who am I in the world?
	int type; // what type of agent am I?
	double travel_vel; // how fast can I move?
	double travel_step; // how far do I move in one time step
	bool pay_obstacle_cost; // do obstacles affect me?
	cv::Scalar color; // what color am I plotted?
	int n_tasks; // how many tasks are there

	// functions
	bool at_node(); // am I at a node, by edge_progress
	bool at_goal(); // am I at my goal node?
	
	// working and planning
	void work_on_task(); // work on the task I am at
	
	// moving and path planning 
	void move_along_edge(); // move along the current edge
	void select_next_edge(); // have a goal, select next edge to get to goal
};

