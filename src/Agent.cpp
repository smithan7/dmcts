#include "Agent.h"
#include "World.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Agent_Planning.h"
#include "Goal.h"
#include "Pose.h"

#include <iostream>
#include <ctime>


// ros stuff
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_broadcaster.h>
#include "angles/angles.h"
#include <custom_messages/DMCTS_Travel_Goal.h>
#include <custom_messages/DMCTS_Probability.h>
#include "custom_messages/Get_Task_List.h"
#include "custom_messages/Complete_Work.h"
#include "custom_messages/Recieve_Agent_Locs.h"



Agent::Agent(ros::NodeHandle nHandle, const int &index_in, const int &type, const double &travel_vel, const cv::Scalar &color, const bool &pay_obstacle_cost, const double &work_radius, const bool &actual_agent, World* world_in, const double &des_alt){
	this->initialized = false;
	this->map_offset_x = 0.0;
	this->map_offset_y = 0.0;
	this->desired_alt = des_alt;
	// am I the actual agent or a dummy agent?
	if(!actual_agent){
		// dummy agent
		this->world = world_in;
		this->index = index_in;
		this->work_radius = work_radius;
		this->type = type;
		this->travel_vel = travel_vel;
		this->pay_obstacle_cost = pay_obstacle_cost;
		this->color = color;
		this->coordinator = new Agent_Coordinator(this, this->world->get_n_nodes());
	}
	else{
		this->world = world_in;
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

		this->index = index_in;
		char bf[50];
		int n = sprintf(bf, "/dmcts_%i/travel_goal", this->index);
		this->move_pub = nHandle.advertise<custom_messages::DMCTS_Travel_Goal>(bf, 10);
		this->coord_pub = nHandle.advertise<custom_messages::DMCTS_Probability>("/dmcts_master/team_coordination", 10);
		ROS_WARN("still depending on world node to publish tasks");
		//this->work_complete_pub = nHandle.advertise<custom_messages::DMCTS_Work_Complete>("/dmcts_master/work_complete", 10);
		n = sprintf(bf, "/uav%i//ground_truth/state", this->index);
		this->odom_sub = nHandle.subscribe(bf, 1, &Agent::odom_callback, this);
		this->coord_sub = nHandle.subscribe("/dmcts_master/team_coordination", 1, &Agent::coord_plan_callback, this);
		this->pulse_sub = nHandle.subscribe("/dmcts_master/pulse", 1, &Agent::pulse_callback, this);

		this->plan_duration = ros::Duration(0.1);
		this->act_duration = ros::Duration(1.0);
		this->send_loc_duration = ros::Duration(1.0);
		this->task_list_duration = ros::Duration(5.0);
		this->plan_timer = nHandle.createTimer(this->plan_duration, &Agent::plan_timer_callback, this);
		this->act_timer = nHandle.createTimer(this->act_duration, &Agent::act_timer_callback, this);
		this->send_loc_timer = nHandle.createTimer(this->send_loc_duration, &Agent::send_loc_service_timer_callback, this);
		this->task_list_timer = nHandle.createTimer(this->task_list_duration, &Agent::task_list_timer_callback, this);

		this->task_list_client = nHandle.serviceClient<custom_messages::Get_Task_List>("/dmcts_master/get_task_list");
		this->send_loc_client = nHandle.serviceClient<custom_messages::Recieve_Agent_Locs>("/dmcts_master/recieve_agent_locs");
		this->work_client = nHandle.serviceClient<custom_messages::Complete_Work>("/dmcts_master/complete_work");

		this->edge.x = -1;
		this->edge.y = -1;
		this->edge_progress = 1.0;
		this->index = index;
		this->collected_reward = 0.0;
		this->run_status = -1;

		this->work_radius = work_radius;
		this->type = type;
		this->travel_vel = travel_vel;
		this->travel_step = travel_vel * world->get_dt();
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

/*void Agent::clock_callback(const rosgraph_msgs::Clock &tmIn){
	this->world->set_time(tmIn.clock.now().toSec());
}*/

void Agent::pulse_callback(const custom_messages::DMCTS_Pulse &msg){
	if(this->index > msg.my_index){
		this->world->set_time(msg.c_time);
	}
	if(this->world->get_n_active_tasks() != msg.n_active_tasks){
		this->send_request_for_task_list();
	}
}

void Agent::publish_coord_plan(){
	custom_messages::DMCTS_Probability msg = custom_messages::DMCTS_Probability();
	msg.agent_index = this->index;
	this->coordinator->get_plan(msg.claimed_tasks, msg.claimed_time, msg.claimed_probability);
	coord_pub.publish(msg);
}

void Agent::coord_plan_callback(const custom_messages::DMCTS_Probability &msg){
	if(this->index == msg.agent_index){
		return;
	}
	//for(size_t i=0; i<msg.claimed_tasks.size(); i++){
	//	ROS_INFO("Agent::coord_plan_callback: agent[%i], task: %i, time: %0.2f, and prob: %0.2f", msg.agent_index, msg.claimed_tasks[i], msg.claimed_time[i], msg.claimed_probability[i]);
	//}
	//ROS_INFO("Agent[%i]'s understanding of agent[%i]s coord tree", this->get_index(), msg.agent_index);
	//this->world->get_agents()[msg.agent_index]->get_coordinator()->print_prob_actions();
	
	this->planner->reset_mcts_team_prob_actions(); // this makes my tree re-check the probability
	this->world->get_agents()[msg.agent_index]->upload_new_plan(msg.claimed_tasks, msg.claimed_time, msg.claimed_probability);
	//this->world->get_agents()[msg.agent_index]->get_coordinator()->print_prob_actions();
	
}

void Agent::upload_new_plan(const std::vector<int> &claimed_tasks, const std::vector<double> &claimed_time, const std::vector<double> &claimed_probability){
	this->coordinator->upload_new_plan(claimed_tasks, claimed_time, claimed_probability);
}

void Agent::task_list_timer_callback(const ros::TimerEvent &e){
	this->send_request_for_task_list();
}

void Agent::plan_timer_callback(const ros::TimerEvent &e){
	if(this->m_node_initialized){
		if(this->task_list_initialized){
			if(!this->plan()){
				ROS_ERROR("Agent[%i]::plan_timer_callback: Agent->plan() Failed", this->index);
			}
			else{
				this->plan_initialized = true;
				this->publish_coord_plan();
			}
		}
		else{
			ROS_WARN("Agent[%i]::plan_timer_callback: task_list_initialized is FALSE", this->index);
			this->send_request_for_task_list();
		}
	}
	else{
		//ROS_WARN("Agent[%i]::plan_timer_callback: m_node_initialized is FALSE", this->index);
	}
}

void Agent::act_timer_callback(const ros::TimerEvent &e){
	if(this->plan_initialized && this->m_node_initialized && this->task_list_initialized){
		if(!this->act()){
			ROS_ERROR("Agent[%i]::act_timer_callback: Agent->act() Failed", this->index);
		}
		else{
			this->act_initialized = true;
		}	
	}
}

void Agent::send_loc_service_timer_callback(const ros::TimerEvent &e){
	//why does this->index iterate 1-> 5 ???
	custom_messages::Recieve_Agent_Locs srv;
	srv.request.index = this->index;
	srv.request.xLoc = this->pose->get_x();
	srv.request.yLoc = this->pose->get_y();
	srv.request.alt = this->pose->get_z();
	srv.request.yaw = this->pose->get_yaw();
	srv.request.edge_x = this->edge.x;
	srv.request.edge_y = this->edge.y;
	srv.request.status = int8_t(this->run_status);

	//ROS_INFO("DMCTS::Agent::send_loc_service_timer_callback: req.status[%i]: %i", this->index, this->run_status);
	//ROS_INFO("   Z: = %0.1f and desired_alt = %0.1f", this->pose->get_z(), this->desired_alt);
	if (this->send_loc_client.call(srv)){
	    if(srv.response.f){
	    	this->m_node_initialized = true;
	    	this->run_status = 1; // all good, run
	    }
	    else{
	    	// problem, wait
	    	this->run_status = 0;
	    }
	}
	else{
		// had a problem, go back to waiting
		this->run_status = 0;
	    ROS_ERROR("Agent[%i]::send_loc_service_timer_callback: Failed to call service: Recieve_Agent_Locs", this->index);
	    return;
	}
}

void Agent::odom_callback(const nav_msgs::Odometry &odom_in){
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
		this->travel_step = this->travel_vel * world->get_dt();
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
	custom_messages::DMCTS_Travel_Goal msg = custom_messages::DMCTS_Travel_Goal();
	msg.x = this->world->get_nodes()[ni]->get_x();
	msg.y = this->world->get_nodes()[ni]->get_y();
	this->move_pub.publish(msg);
}

void Agent::publish_work_request(const int &goal_node ){
	custom_messages::Complete_Work srv;
	srv.request.n_index = goal_node;
	srv.request.xLoc = this->pose->get_x();
	srv.request.yLoc = this->pose->get_y();
	srv.request.a_type = this->type;
	srv.request.c_time = this->world->get_c_time();
	srv.request.work_rate = this->act_duration.toSec();
	if (this->work_client.call(srv)){
	    int a;
	}
	else{
	    ROS_ERROR("Agent[%i]::publish_work_request: Failed to call service Complete_Work", this->index);
	    return;
	}	
}

void Agent::send_request_for_task_list(){
	custom_messages::Get_Task_List srv;
	srv.request.f = true;
	//ROS_ERROR("Agent[%i]::send_request_for_task_list: sent");
	if (this->task_list_client.call(srv)){
		//ROS_WARN("Agent[%i]::send_request_for_task_list: recieved");
		//ROS_WARN("Agent[%i]::send_request_for_task_list: %i", int(srv.response.node_indices.size()));
		//ROS_WARN("Agent[%i]::send_request_for_task_list: %i", int(srv.response.xLoc.size()));
		//ROS_WARN("Agent[%i]::send_request_for_task_list: %i", int(srv.response.yLoc.size()));
		//ROS_WARN("Agent[%i]::send_request_for_task_list: %i", int(srv.response.reward.size()));

		// check if any currently active tasks need to be deactivated
	    for(size_t i=0; i<this->world->get_nodes().size(); i++){
	   		if(this->world->get_nodes()[i]->is_active()){
	   			bool flag = true;
	   			for(size_t j=0; j<srv.response.node_indices.size(); j++){
	   				if(this->world->get_nodes()[i]->get_index() == srv.response.node_indices[j]){
	   					flag = false;
	   					break;
	   				}
	   			}
	   			// not in list, deactivate
	   			if(flag){
	   				this->world->deactivate_task(i);
	   			}
	   		}
	   	}
	   	// activate all nodes that are not currently active
	    for(size_t i=0; i<srv.response.node_indices.size(); i++){
	   		if(!this->world->get_nodes()[srv.response.node_indices[i]]->is_active()){
	   			this->world->activate_task(srv.response.node_indices[i]);	
	   		}
	    }
	    this->world->reset_task_status_list();
	    this->task_list_initialized = true;
	    
	}
	else{
	    ROS_ERROR("Agent[%i]::send_request_for_task_list: Failed to call service get_task_list", this->index);
	    return;
	}
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
	if(dist <= this->work_radius){
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

void Agent::select_next_edge() {

	std::vector<int> path;
	double length = 0.0;

	this->edge_progress = 0.0;
	if ( this->world->a_star(this->edge.x, this->goal_node->get_index(), this->pay_obstacle_cost, path, length)) {
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
		//ROS_INFO("Agent[%i]::plan: this->edge.x: %i", this->index, this->edge.x);
		this->planner->plan(); // I am not at my goal, select new goal
		//ROS_INFO("Agent[%i]::plan: out of planner->plan", this->index);
		this->coordinator->advertise_task_claim(this->world); // select the next edge on the path to goal
		//ROS_INFO("Agent[%i]::plan: out of advertise_task_claim", this->index);
		return true;
	}
	else{
		ROS_WARN("Agent[%i]::plan: Agent location is not initialized", this->index);
		return true;
	}
}

bool Agent::act() {
	if(this->run_status != 1){
		if(this->run_status == 0){
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
	//ROS_WARN("At node: %i", this->edge.x);
	bool an = false;
	if (this->at_node(this->edge.x)){
		// if my current node is active
		if(this->world->get_nodes()[this->edge.x]->is_active()){
			this->publish_work_request(this->edge.x); // send service to request work
			this->work_done +=  this->world->get_nodes()[this->edge.x]->get_acted_upon(this); // work on my goal
		}
		an = true;
	}
	if(this->at_node(this->edge.y)) {
		an = true;
		// if my current node is active
		if(this->world->get_nodes()[this->edge.y]->is_active()){
			this->publish_work_request(this->edge.y); // send service to request work
			this->work_done +=  this->world->get_nodes()[this->edge.y]->get_acted_upon(this); // work on my goal
		}
	}

	if(an){
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