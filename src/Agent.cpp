#include "Agent.h"
#include "World.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Agent_Planning.h"
#include "Goal.h"
#include "Pose.h"
#include "Distributed_MCTS.h"

#include <iostream>
#include <ctime>


// ros stuff
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_broadcaster.h>
#include "angles/angles.h"

Agent::Agent(ros::NodeHandle nHandle, const int &index_in, const int &type, const double &travel_vel, const cv::Scalar &color, const bool &pay_obstacle_cost, const double &work_radius, const bool &actual_agent, World* world_in, const double &des_alt){
	this->initialized = false;
	this->map_offset_x = 0.0;
	this->map_offset_y = 0.0;
	this->last_pulse_time = ros::Time::now();
	this->pulse_duration = ros::Duration(3.0);
	this->location_radius = world_in->way_point_tollerance;
	// am I the actual agent or a dummy agent?

	this->world = world_in;
	this->index = this->world->my_agent_index;
	this->work_radius = work_radius;
	this->type = type;
	this->travel_vel = this->world->agent_cruising_speed;
	this->pay_obstacle_cost = pay_obstacle_cost;
	this->desired_alt = this->world->desired_alt;

	if(!actual_agent){
		// dummy agent
		this->color = color;
		this->coordinator = new Agent_Coordinator(this, this->world->get_n_nodes());
	}
	else{
		////////////////////// How do I select my goal ////////////////////////////////////////////
		{
			//////////////////////////////// greedy methods
			{
				//this->task_selection_method = "greedy_arrival_time"; // choose the closest active task
				//this->task_selection_method = "greedy_completion_time"; // choose the task I can finish first
				
				//this->task_selection_method = "greedy_current_reward"; // choose the task with the largest reward currently
				//this->task_selection_method = "greedy_arrival_reward"; // choose the task with the largest reward at the time I will arrive
				//this->task_selection_method = "greedy_completion_reward"; // choose the task with the largest reward at the time I will complete
			}
			//////////////////////////////// value methods
			{
				//this->task_selection_method = "value_current"; // choose task by value now, value = reward(t_current) - (travel_time + work_time)
				//this->task_selection_method = "value_arrival"; // choose task by value at time of arrival, value = reward(t_arrival) - (travel_time + work_time)
				//this->task_selection_method = "value_completion"; // choose task by value at time of completion, value = reward(t_complete) - (travel_time + work_time)
			}
			//////////////////////////////// impact methods
			{
				//this->task_selection_method = "impact_completion_reward"; // choose task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete)
				//this->task_selection_method = "impact_completion_value"; // choose task by impact at value time of completion, impact_value = reward(t_complete) - reward(t^{next closest agent}_complete) - (travel_time + work_time)
			}


			//////////////////////////////// random methods
			{
				//this->task_selection_method = "random_nbr"; // choose a random nbr
				//this->task_selection_method = "random_node"; // choose a random node on the map
				//this->task_selection_method = "random_task"; // choose a random active task
			}
			///////////////////////////////// MCTS methods
			{
				//this->task_selection_method = "mcts_task_by_completion_reward";
				//this->task_selection_method = "mcts_task_by_completion_value"; // use MCTS to plan a sequence of values 
				//this->task_selection_method = "mcts_task_by_completion_reward_impact"; //
				//this->task_selection_method = "mcts_task_by_completion_value_impact";

			}
			this->task_selection_method =  this->world->get_task_selection_method();
		}
		//////////////////////// When do I claim my tasks /////////////////////////////////////////
		{
			this->task_claim_time = "completion_time"; // when I will complete the task
			//this->task_claim_time = "arrival_time"; // when I will arrive at the task
			//this->task_claim_time = "immediate"; // I claim it from right now
			//this->task_claim_time = "none"; // I do NOT claim the task at all
		}
		/////////////////////// How do I claim my tasks ///////////////////////////////////////////
		{
			this->task_claim_method = "greedy"; // whatever task is best gets P(t) = 1.0, else P(t) = 0.0;
			//this->task_claim_method = "sample"; // all tasks get P(t) = (V(t)-V_min)/(V_max-V_min);
		}

		// Publish to Quad
		char bf[50];
		int n = sprintf(bf, "/dmcts_%i/travel_goal", this->index);
		this->move_pub = nHandle.advertise<custom_messages::DMCTS_Travel_Goal>(bf, 10);

		// Subscribe to Quad
		n = sprintf(bf, "/uav%i//ground_truth/state", this->index);
		this->odom_sub = nHandle.subscribe(bf, 1, &Agent::odom_callback, this);
		
		// Subscribe to topics via Xbee
		this->coord_sub = nHandle.subscribe("/dmcts_master/coordination", 10, &Agent::coordination_callback, this);
		this->pulse_sub = nHandle.subscribe("/dmcts_master/pulse", 1, &Agent::pulse_callback, this);
		this->task_list_sub = nHandle.subscribe("/dmcts_master/task_list", 5, &Agent::task_list_callback, this);
		this->work_status_sub = nHandle.subscribe("/dmcts_master/work_status", 10, &Agent::work_status_callback, this);

		// Publish to Topics via XBee
		this->coord_pub = nHandle.advertise<custom_messages::DMCTS_Coordination>("/dmcts_master/coordination", 10);
		this->loc_pub = nHandle.advertise<custom_messages::DMCTS_Loc>("/dmcts_master/loc", 10);
		this->request_task_list_pub = nHandle.advertise<custom_messages::DMCTS_Request_Task_List>("/dmcts_master/request_task_list", 10);
		this->request_work_pub = nHandle.advertise<custom_messages::DMCTS_Request_Work>("/dmcts_master/request_work", 10);

		// Timer Durations
		this->plan_duration = ros::Duration(0.5);
		if(this->index == 0){
			this->act_duration = ros::Duration(0.1);
		}
		else{
			this->act_duration = ros::Duration(0.5);
		}
		this->send_loc_duration = ros::Duration(1.0);
		
		this->task_list_timer_duration = ros::Duration(0.5); // How frequently do I check if I need to update the task list
		this->task_list_wait_duration = ros::Duration(3.0); // Wait for reply before resending request
		this->task_list_request_sent_time = ros::Time::now();
		
		this->work_request_timer_duration = ros::Duration(0.1); // Check if I need to resend the task list
		this->work_wait_duration = ros::Duration(1.0); // Wait 3 seconds before sending a second request for work
		this->work_request_sent_time = ros::Time::now();

		// Timer Callbacks
		this->plan_timer = nHandle.createTimer(this->plan_duration, &Agent::plan_timer_callback, this);
		this->act_timer = nHandle.createTimer(this->act_duration, &Agent::act_timer_callback, this);
		this->send_loc_timer = nHandle.createTimer(this->send_loc_duration, &Agent::publish_loc_timer_callback, this);
		this->task_list_timer = nHandle.createTimer(this->task_list_timer_duration, &Agent::task_list_timer_callback, this);
		this->work_timer = nHandle.createTimer(this->work_request_timer_duration, &Agent::work_timer_callback, this);

		this->waiting_on_work_status = false;
		this->waiting_on_task_list = false;

		this->edge.x = -1;
		this->edge.y = -1;
		this->edge_progress = 1.0;
		this->index = index;
		this->collected_reward = 0.0;
		this->run_status = -1;

		this->work_radius = work_radius;
		this->type = type;
		this->travel_vel = travel_vel;
		this->pay_obstacle_cost = pay_obstacle_cost;
		this->color = color;
		this->n_tasks =  this->world->get_n_nodes();

		this->goal_node = new Goal();
		this->planner = new Agent_Planning(this, world);
		this->coordinator = new Agent_Coordinator(this, n_tasks);
		this->pose = new Pose(-1,-1,-1,0);
		this->task_list_initialized = false;
		this->m_node_initialized = false;
		this->plan_initialized = false;
		this->act_initialized = false;
		this->initialized = true;
	}
}

bool Agent::get_at_node(const int &node){
	return this->at_node(node);
}

void Agent::task_list_timer_callback(const ros::TimerEvent &e){
	if(this->waiting_on_task_list == false){
		return;
	}

	if(ros::Time::now() - this->task_list_request_sent_time > this->task_list_wait_duration){
		this->publish_task_list_request();
	}
}

void Agent::work_timer_callback(const ros::TimerEvent &e){
	if(this->waiting_on_work_status == false){
		return;
	}

	if(ros::Time::now() - this->work_request_sent_time > this->work_wait_duration){
		this->publish_work_request(this->goal_node->get_index());
	}
}

void Agent::pulse_callback(const custom_messages::DMCTS_Pulse &msg){
	//ROS_INFO("Agent::pulse_callback: in ");
	this->last_pulse_time = ros::Time::now();
	if(this->index > msg.my_index){
		this->world->set_time(msg.c_time);
	}
	if(msg.status == 1){
    	this->m_node_initialized = true;
    	this->run_status = 1; // all good, run
    }
    else{
    	if(this->m_node_initialized){
			// had a problem, go back to waiting
			this->run_status = 0;
		    ROS_ERROR("Agent[%i]::pulse_callback: Ground Station has an Error: DMCTS_Loc", this->index);
		    return;
    	}
    	else{
	    	// problem, wait
	    	ROS_WARN("DMCTS::Agent[%i]::pulse_callback: Waiting to run (%i)", this->index, this->run_status);
	    	this->run_status = 0;
    	}
	}
	if(this->world->get_n_active_tasks() != msg.n_active_tasks){
		this->request_task_list();
		this->waiting_on_task_list = true;
	}
}

void Agent::publish_coordination(){
	custom_messages::DMCTS_Coordination msg = custom_messages::DMCTS_Coordination();
	msg.agent_index = this->index;
	this->coordinator->get_plan(msg.claimed_tasks, msg.claimed_time, msg.claimed_probability);
	coord_pub.publish(msg);
}

void Agent::coordination_callback(const custom_messages::DMCTS_Coordination &msg){
	//ROS_WARN("my index is %i and msg.index is %i", this->index, msg.agent_index);
	//ROS_INFO("Agent::coordination_callback: in");
	if(this->index == msg.agent_index){
	//	ROS_WARN("returning");
		return;
	}
	//ROS_WARN("still here");
	//for(size_t i=0; i<msg.claimed_tasks.size(); i++){
	//	ROS_INFO("Agent[%i]::coord_plan_callback: agent[%i], task: %i, time: %0.2f, and prob: %0.2f", this->index, msg.agent_index, msg.claimed_tasks[i], msg.claimed_time[i], msg.claimed_probability[i]);
	//}
	//ROS_INFO("Agent[%i]'s understanding of agent[%i]s coord tree", this->get_index(), msg.agent_index);
	//this->world->get_agents()[msg.agent_index]->get_coordinator()->print_prob_actions();
	
	if(msg.agent_index < int(this->world->get_agents().size())){
    	this->world->get_agents()[msg.agent_index]->upload_new_plan(msg.claimed_tasks, msg.claimed_time, msg.claimed_probability);
    }
	//this->world->get_agents()[msg.agent_index]->get_coordinator()->print_prob_actions();
	
}

void Agent::upload_new_plan(const std::vector<int> &claimed_tasks, const std::vector<double> &claimed_time, const std::vector<double> &claimed_probability){
	this->coordinator->upload_new_plan(claimed_tasks, claimed_time, claimed_probability);
}

void Agent::publish_task_list_request(){
	custom_messages::DMCTS_Request_Task_List msg;
	msg.request = 1; // This can literally be anything...
	this->request_task_list_pub.publish(msg);
	this->task_list_request_sent_time = ros::Time::now();	
}

void Agent::plan_timer_callback(const ros::TimerEvent &e){
	//ROS_INFO("Agent::plan_timer_callback: in");
	if(this->m_node_initialized){
		if(this->task_list_initialized){
			if(!this->plan()){
				ROS_ERROR("Agent[%i]::plan_timer_callback: Agent->plan() Failed", this->index);
			}
			else{
				this->plan_initialized = true;
				this->publish_coordination();
			}
		}
		else{
			ROS_WARN("Agent[%i]::plan_timer_callback: task_list_initialized is FALSE", this->index);
			this->publish_task_list_request();
		}
	}
	else{
		//ROS_WARN("Agent[%i]::plan_timer_callback: m_node_initialized is FALSE", this->index);
	}
}

void Agent::act_timer_callback(const ros::TimerEvent &e){
	//ROS_INFO("Agent::act_timer_callback: in");
	if(this->plan_initialized && this->m_node_initialized && this->task_list_initialized){
		if(!this->act()){
			ROS_ERROR("Agent[%i]::act_timer_callback: Agent->act() Failed", this->index);
		}
		else{
			this->act_initialized = true;
		}	
	}
	//cv::Mat emp(cv::Size(10, 10), CV_8UC1);
	//cv::namedWindow("na", cv::WINDOW_NORMAL);
	//cv::imshow("na", emp);
	//cv::waitKey(0);
}

void Agent::publish_loc_timer_callback(const ros::TimerEvent &e){
	//ROS_INFO("Agent::publish_loc_timer_callback: in");
	
	if (ros::Time::now() - this->last_pulse_time > this->pulse_duration){
		this->run_status = 0;
		ROS_WARN("Agent::publish_loc_timer_callback::Have NOT heard pulse from Groundstation, switching modes");
	}


	custom_messages::DMCTS_Loc msg;
	msg.index = this->index;
	msg.xLoc = this->pose->get_x();
	msg.yLoc = this->pose->get_y();
	msg.edge_x = this->edge.x;
	msg.edge_y = this->edge.y;
	msg.status = int8_t(this->run_status);
	msg.path = this->get_path();

	this->loc_pub.publish(msg);
}

void Agent::odom_callback(const nav_msgs::Odometry &odom_in){
	//ROS_INFO("Agent::odom_callback: in");
	// from video https://www.youtube.com/watch?v=LDMybJQVohk
	double r,p,yaw;
	tf::Quaternion quater;
	tf::quaternionMsgToTF(odom_in.pose.pose.orientation, quater);
	tf::Matrix3x3(quater).getRPY(r,p,yaw);
	yaw = angles::normalize_angle_positive(yaw);
	// end video notes on "Convert quaternion to euler in C++ ROS"
	//ROS_ERROR("vel: %0.2f, %0.2f", odom_in.twist.twist.linear.x, odom_in.twist.twist.linear.y);

	if(abs(odom_in.pose.pose.position.z - this->desired_alt) < 1.0){
		double ts = sqrt(pow(odom_in.twist.twist.linear.x,2) + pow(odom_in.twist.twist.linear.y,2));
		this->travel_vel = this->travel_vel + 0.001 * (ts - this->travel_vel);
		//ROS_INFO("travel_vel: %0.2f", this->travel_vel);
		if(this->run_status == -1){
			this->run_status = 0; // I have reached altitude
		}
	}
	else{ // I am no longer at altitude
		this->run_status = -1;
	}

	this->update_pose(odom_in.pose.pose.position.x, odom_in.pose.pose.position.y, odom_in.pose.pose.position.z, yaw);
	//ROS_WARN("Agent[%i]::odom_callback::odom_in: %.2f, %.2f", this->pose->get_x(), this->pose->get_y());

	// have I set my starting pose?
	if(!this->location_initialized){
		// have I initialized myself?
		if(this->initialized){
			double min_dist = INFINITY;
			int mindex = -1;
			for(int i=0; i<this->world->get_n_nodes(); i++){
				double dist = sqrt(pow(this->pose->get_x() - this->world->get_nodes()[i]->get_x(),2) + pow(this->pose->get_y() - this->world->get_nodes()[i]->get_y(),2));
				if(dist < min_dist){
					mindex = i;
					min_dist = dist;
				}
			}
			this->edge.x = mindex;
			this->edge.y = mindex;
			this->goal_node->set_index(mindex);
			this->location_initialized = true;
		}
	}	
	else{ // I have been initialized, check if my edge should be updated
		double dist_to_edge_y = sqrt( pow(this->world->get_nodes()[this->edge.y]->get_x() - this->pose->get_x(),2) + pow(this->world->get_nodes()[this->edge.y]->get_y() - this->pose->get_y(),2) );
		double cost = INFINITY;
		this->world->get_edge_cost(this->edge.x, this->edge.y, this->pay_obstacle_cost, cost);
		if(cost == 0){
			this->edge_progress = 1.0;
			this->edge.x = this->edge.y;
		}
		else{
			this->edge_progress = 1 - (dist_to_edge_y / cost);
		}

		if(this->at_node(this->edge.y)){
			this->edge.x = this->edge.y;
			this->select_next_edge();	
		}
	}
}

void Agent::update_pose(const double &xi, const double &yi, const double &zi, const double wi){
	this->pose->update_pose(xi + this->map_offset_x, yi + this->map_offset_y, zi, wi);
}

void Agent::publish_to_control_script(const int &ni){
	custom_messages::DMCTS_Travel_Goal msg;
	msg.x = this->world->get_nodes()[ni]->get_x();
	msg.y = this->world->get_nodes()[ni]->get_y();
	this->move_pub.publish(msg);
}

void Agent::publish_work_request(const int &goal_node ){
	custom_messages::DMCTS_Request_Work msg;
	msg.n_index = goal_node;
	msg.a_type = this->type;
	msg.a_index = this->index;
	//ROS_WARN("Agent::publish_work_request: my agent_index: %i", msg.a_type);
	this->request_work_pub.publish(msg);
}

void Agent::request_task_list(){
	//ROS_ERROR("Agent[%i]::request_task_list: sent");
	custom_messages::DMCTS_Request_Task_List msg;
	msg.request = 1;
	this->request_task_list_pub.publish(msg);
}

void Agent::work_status_callback(const custom_messages::DMCTS_Work_Status &msg){
	//ROS_INFO("Agent::work_status_callback: in");
	if (msg.success == 0){
		this->world->get_nodes()[msg.n_index]->deactivate();
		return;
	}


	if(msg.a_index == this->index){
		// is it me
		if(msg.success == 1){
			// I did some work, but not complete
			ROS_INFO("Agent::work_status_callback: I worked on node %i", msg.n_index);
		}
		if(msg.success == -1){
			// I failed to do work
			ROS_WARN("Agent::work_status_callback: I Failed to do work on node %i", msg.n_index);
		}
	}
	this->waiting_on_work_status = false;
}

void Agent::task_list_callback(const custom_messages::DMCTS_Task_List &msg){
	//ROS_INFO("Agent::task_list_callback: in");
	//ROS_WARN("Agent[%i]::task_list_callback: recieved");
	//ROS_WARN("Agent[%i]::task_list_callback: %i", int(srv.response.node_indices.size()));
	
	// check if any currently active tasks need to be deactivated
    for(size_t i=0; i<this->world->get_nodes().size(); i++){
   		if(this->world->get_nodes()[i]->is_active()){
   			bool flag = true;
   			for(size_t j=0; j<msg.node_indices.size(); j++){
   				if(this->world->get_nodes()[i]->get_index() == msg.node_indices[j]){
   					flag = false;
   					break;
   				}
   			}
   			// not in list, deactivate
   			if(flag){
   				this->world->deactivate_task(i);
   				if(this->planner->get_dist_mcts()){
   					this->planner->get_dist_mcts()->clean_task(i);
   				}
   			}
   		}
   	}
   	// activate all nodes that are not currently active
    for(size_t i=0; i<msg.node_indices.size(); i++){
   		if(!this->world->get_nodes()[msg.node_indices[i]]->is_active()){
   			this->world->activate_task(msg.node_indices[i]);	
   		}
    }
    this->world->reset_task_status_list();
    this->task_list_initialized = true;
    this->waiting_on_task_list = false;    
}

bool Agent::at_node(int node) {
	//ROS_ERROR("this->pose: %.2f, %.2f", this->pose->get_x(), this->pose->get_y());
	//ROS_ERROR("world::node loc: %.2f, %.2f", this->world->get_nodes()[node]->get_x(), this->world->get_nodes()[node]->get_y());
	if(node < 0 || node >= this->world->get_nodes().size()){
		return false;
	}
	double dx = this->pose->get_x() - this->world->get_nodes()[node]->get_x();
	double dy = this->pose->get_y() - this->world->get_nodes()[node]->get_y();
	//ROS_ERROR("dx/dy: %.2f, %.2f", dx, dy);
	double dist = sqrt(pow(dx,2) + pow(dy,2));
	//ROS_ERROR("dist: %0.2f", dist);
	if(dist <= this->location_radius){
		return true;
	}
	else{
		return false;
	}
}

cv::Point2d Agent::get_loc2d() {
	cv::Point2d p(0.0, 0.0);
	p.x = ( this->world->get_nodes()[this->edge.y]->get_x() -  this->world->get_nodes()[this->edge.x]->get_x())*this->edge_progress +  this->world->get_nodes()[this->edge.x]->get_x();
	p.y = ( this->world->get_nodes()[this->edge.y]->get_y() -  this->world->get_nodes()[this->edge.x]->get_y())*this->edge_progress +  this->world->get_nodes()[this->edge.x]->get_y();
	return p;
}

void Agent::work_on_task() {
	this->work_done +=  this->world->get_nodes()[this->goal_node->get_index()]->get_acted_upon(this);
}

bool Agent::get_travel_time(const int &ti, double &travel_time){

	// this has two parts
		// - First: Get the travel time from current location to edge.y
		// - Second: Get the travel time from edge.y to node[ti]
		// travel_time = 1 + 2

	// First: get travel time from current location to edge.y
	double node_dist = this->pose->distance_to(this->world->get_nodes()[this->edge.y]);

	// Second: A* path from edge.y to node[ti]
	std::vector<int> path;
	double path_length = 0.0;
	bool need_path = false;
	if ( this->world->a_star(this->edge.y, this->world->get_nodes()[ti]->get_index(), this->pay_obstacle_cost, need_path, path, path_length)){
		double travel_distance = node_dist + path_length;
		travel_time = travel_distance / this->travel_vel;
		return true;
	}
	else{
		travel_time = INFINITY;
		return false;
	}
}

void Agent::select_next_edge() {

	if(this->planner->get_dist_mcts()){
		this->planner->Distributed_MCTS_exploit_tree();
	}

	std::vector<int> path;
	double length = 0.0;
	bool need_path = true;
	
	this->edge_progress = 0.0;
	if ( this->world->a_star(this->edge.x, this->goal_node->get_index(), this->pay_obstacle_cost, need_path, path, length)) {
		if (path.size() >= 2) {
			this->edge.y = path.end()[-2];
		}
	}
	else {
		ROS_ERROR("Agent[%i]::select_next_edge::a_star failed to find path", this->index);
	}
}

bool Agent::at_node() { // am I at a node, by edge progress?
	if (this->at_node(this->edge.x) || this->at_node(this->edge.y)) {
		return true;
	}
	else {
		return false;
	}
}

bool Agent::at_goal() { // am I at my goal node?
	if (this->at_node(this->goal_node->get_index())) {
		return true;
	}
	else {
		return false;
	}
}

bool Agent::plan(){
	//ROS_INFO("Agent[%i]::plan: in", this->index);
	if(this->location_initialized){
		//ROS_INFO("Agent[%i]::plan: this->edge: %i -> %i", this->index, int(this->edge.x), int(this->edge.y));
		this->planner->plan(); // Figure out where to go
		//ROS_INFO("Agent[%i]::plan: out of planner->plan", this->index);
		//ROS_INFO("Agent[%i]::plan: on edge %i -> %i", this->index, int(this->edge.x), int(this->edge.y));
		this->coordinator->advertise_task_claim(this->world); // Advertise where I am going
		//ROS_INFO("Agent[%i]::plan: out of advertise_task_claim", this->index);
		//ROS_INFO("Agent[%i]::plan: on edge %i -> %i", this->index, int(this->edge.x), int(this->edge.y));
		return true;
	}
	else{
		ROS_WARN("Agent[%i]::plan: Agent location is not initialized", this->index);
		return true;
	}
}

bool Agent::act() {
	if(this->run_status < 1){
		if(this->run_status == -1){
			ROS_WARN("DMCTS::Agent::act: %i is not ready to run: alt info: %0.1f / %0.1f", this->index, this->pose->get_z(), this->desired_alt);
		}
		else if(this->run_status == 0){
			ROS_WARN("DMCTS::Agent::act: %i is waiting to at travel altitude", this->index);	
		}
		else{
			ROS_ERROR("DMCTS::Agent::act: BAD run_status");
		}
		return true;
	}

	//ROS_INFO("Agent[%i]::act: in", this->index);
	// am  I at a node?
	//ROS_WARN("Agent[%i]::act: on edge: %i -> %i", this->index, this->edge.x, this->edge.y);

	if(this->at_node(this->edge.y) || this->at_node(this->edge.x)){
		//ROS_INFO("Agent[%i]::act: at node", this->index);
		// I am at a node, am I at my goal and is it active still?
		if (this->at_node(this->goal_node->get_index()) && this->world->get_nodes()[this->goal_node->get_index()]->is_active()) {
			//ROS_INFO("Agent[%i]::act: at goal, trying to work", this->index);
			this->publish_work_request(this->goal_node->get_index()); // send service to request work
			this->work_done +=  this->world->get_nodes()[this->goal_node->get_index()]->get_acted_upon(this); // work on my goal
		}
		else {
			//ROS_INFO("Agent[%i]::act: not at goal, selecting next edge and moving", this->index);
			this->select_next_edge(); // plan the next edge
			this->move_along_edge(); // on the right edge, move along edge
		}
	}
	else { // not at a node
		//ROS_INFO("Agent[%i]::act: not at node, moving along edge towards node %i", this->index, this->edge.y);
		this->move_along_edge(); // move along edge
	}
	return true;
}

void Agent::move_along_edge() {
	// how long is the current edge?
	if(this->edge.x < 0 || this->edge.y < 0 || this->edge.x >= this->world->get_nodes().size() || this->edge.x >= this->world->get_nodes().size()){
		ROS_WARN("Agent[%i]::move_along_edge: invalid edge (x/y): %i/ %i", this->index, this->edge.x, this->edge.y );
		return;
	}
	double cost = 0;
	if (this->world->get_edge_cost(this->edge.x, this->edge.y, this->pay_obstacle_cost, cost) || this->edge.x == this->edge.y) {
		if (cost == 0 && this->at_node(this->edge.y)) {
			this->edge_progress = 1.0;
			this->edge.x = this->edge.y;
		}
		else {
			//ROS_INFO("Agent[%i]::move_along_edge: publish_to_control_script %i", this->index, this->edge.y);
			//ROS_ERROR("Agent::move_along_edge: disabled movement");
			this->publish_to_control_script(this->edge.y);
		}
	}
	else {
		ROS_ERROR("Agent[%i]::move_along_edge::bad request", this->index);
	}
}

Agent::~Agent(){
	delete this->planner;
	delete this->coordinator;
	delete this->goal_node;
	delete this->pose;
}
