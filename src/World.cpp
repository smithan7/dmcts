#include "World.h"

#include "Map_Node.h"
#include "Agent.h"
#include "Proabability_Node.h"
#include "Agent_Coordinator.h"
#include "Agent_Planning.h"
#include "Goal.h"
#include "Pose.h"

#include <random>
#include <iostream>
#include <fstream>

// opencv stuff
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

World::World(ros::NodeHandle nHandle){

	double speed_penalty;
	ros::param::get("/test_environment_img", this->test_environment_img);
	ros::param::get("/test_obstacle_img", this->test_obstacle_img);
	ros::param::get("/agent_index", this->my_agent_index);
	ros::param::get("/world_directory", this->world_directory);
	ros::param::get("/score_run", this->score_run);
	ros::param::get("/param_number", this->rand_seed);
	ros::param::get("/number_of_nodes", this->n_nodes);
	ros::param::get("/number_of_agents", this->n_agents);
	ros::param::get("/coord_method", this->task_selection_method);
	ros::param::get("/desired_altitude", this->desired_alt);
	ros::param::get("/p_task_initially_active", this->p_task_initially_active);
	ros::param::get("/pay_obstacle_costs", this->pay_obstacle_cost);
	ros::param::get("/cruising_speed", this->agent_cruising_speed);
	ros::param::get("/use_gazebo", this->use_gazebo);
	ros::param::get("/alpha", this->alpha);
	ros::param::get("/beta", this->beta);
	ros::param::get("/gamma", this->gamma);
	ros::param::get("/epsilon", this->epsilon);
	ros::param::get("/min_sampling_threshold", this->min_sampling_threshold);
	ros::param::get("/search_depth", this->search_depth);
	ros::param::get("/end_time", this->end_time);
	ros::param::get("/way_point_tol", this->way_point_tollerance);
	ros::param::get("/north_lat", this->north_lat);
	ros::param::get("/south_lat", this->south_lat);
	ros::param::get("/east_lon", this->east_lon);
	ros::param::get("/west_lon", this->west_lon);
	ros::param::get("/inflation_iters", this->inflation_iters);
	ros::param::get("/obstacle_increase", this->obstacle_increase);
	ros::param::get("/agent_display_map", this->show_display);
	ros::param::get("/hardware_trial", this->hardware_trial);
	ros::param::get("/agent_type", this->my_agent_type);
	ros::param::get("/flat_tasks", this->flat_tasks);
	ros::param::get("/speed_penalty", speed_penalty);
	ros::param::get("/n_task_types", this->n_task_types);
	ros::param::get("/n_agent_types", this->n_agent_types);
	
   	this->test_obstacle_img = this->world_directory + this->test_obstacle_img;
    this->test_environment_img = this->world_directory + this->test_environment_img;
    this->world_directory = this->world_directory + "/worlds/";

	ROS_INFO("World::initializing agent's world");
	ROS_INFO("   test_environment_img %s", this->test_environment_img.c_str());
	ROS_INFO("   test_obstacle_img %s", this->test_obstacle_img.c_str());
	ROS_INFO("   agent_index %i", this->my_agent_index);
	ROS_INFO("   world directory %s", this->world_directory.c_str());
	ROS_INFO("   score_run %i", this->score_run);
	ROS_INFO("   display_map %i", this->show_display);
	ROS_INFO("   parameter_seed %i", this->rand_seed);
	ROS_INFO("   n_nodes %i", this->n_nodes);
	ROS_INFO("   n_agents %i", this->n_agents);
	ROS_INFO("   coord_method %s", this->task_selection_method.c_str());
	ROS_INFO("   desired_altitude %.2f", this->desired_alt);
	ROS_INFO("   p_task_initially_active %0.4f", this->p_task_initially_active);
	ROS_INFO("   pay_obstacle_costs %i", this->pay_obstacle_cost);
	ROS_INFO("   cruising_speed %0.2f", this->agent_cruising_speed);
	ROS_INFO("   use_gazebo %i", this->use_gazebo);
	ROS_INFO("   alpha %0.2f", this->alpha);
	ROS_INFO("   beta %0.2f", this->beta);
	ROS_INFO("   epsilon %0.2f", this->epsilon);
	ROS_INFO("   gamma %0.2f", this->gamma);
	ROS_INFO("   min_sampling_threshold %0.2f", this->min_sampling_threshold);
	ROS_INFO("   search_depth %i", this->search_depth);
	ROS_INFO("   end_time %0.2f", this->end_time);
	ROS_INFO("   way_point_tol %0.2f", this->way_point_tollerance);
	ROS_INFO("   north_lat %0.6f", this->north_lat);
	ROS_INFO("   south_lat %0.6f", this->south_lat);
	ROS_INFO("   west_lon %0.6f", this->west_lon);
	ROS_INFO("   east_lon %0.6f", this->east_lon);
	ROS_INFO("   origin_lat %0.6f", (this->north_lat + this->south_lat)/2.0);
	ROS_INFO("   origin_lon %0.6f", (this->west_lon + this->east_lon)/2.0);
	ROS_INFO("   inflation_iters %i", this->inflation_iters);
	ROS_INFO("   obstacle_increase %0.2f", this->obstacle_increase);
	ROS_INFO("   show display %i", this->show_display);
	ROS_INFO("   hardware_trial %i", this->hardware_trial);
	ROS_INFO("   agent_type %i", this->my_agent_type);
	ROS_INFO("   flat_tasks %i", this->flat_tasks);
	ROS_INFO("   speed_penalty %.2f", speed_penalty);
	ROS_INFO("   n_task_types %i", this->n_task_types);
	ROS_INFO("   n_agent_types %i", this->n_agent_types);

	this->agent_cruising_speed *= (1.0-speed_penalty);

	this->initialized = false;
	this->mcts_search_type = "SW-UCT"; // UCT or SW-UCT
	this->mcts_reward_type = "normal"; // "impact";
	this->impact_style = "nn";
	this->mcts_n_kids = 10;

	// how often do I plot
	this->plot_duration = ros::Duration(1); 
	this->plot_timer = nHandle.createTimer(this->plot_duration, &World::plot_timer_callback, this);

	ROS_INFO("   Recieved RAND_SEED from param server: %i", this->rand_seed);


	// time stuff
	this->c_time = 0.0;
	this->end_time = end_time;

	if(this->test_obstacle_img.empty()){
    	this->map_width_meters = 100.0;
	    this->map_height_meters = 100.0;
	}
	else{
	this->map_width_meters = this->get_global_distance(this->north_lat, this->west_lon, this->north_lat, this->east_lon);
	    this->map_height_meters = this->get_global_distance(this->north_lat, this->west_lon, this->south_lat, this->west_lon);
	}

	if(this->use_gazebo){
		if(this->map_width_meters > this->map_height_meters){
			this->map_height_meters = 90.0 * this->map_height_meters / this->map_width_meters;
			this->map_width_meters = 90.0;
		}
		else{
			this->map_width_meters = 90.0 * this->map_width_meters / this->map_height_meters;
			this->map_height_meters = 90.0;
		}
	}
    ROS_INFO("DMCTS_world_node::   Word::World(): map size: %0.2f, %0.2f (meters)", this->map_width_meters, this->map_height_meters);
	this->n_obstacles = 10;
	this->k_map_connections = 5;
	this->k_connection_radius = 10.0;
	this->p_connect = 1.0;
	this->p_blocked_edge = 0.05;
	this->p_obstacle_on_edge = 0.2;
	this->p_pay_obstacle_cost = 0.0;
	// task stuff
	this->p_impossible_task = 0.0; // how likely is it that an agent is created that cannot complete a task
	this->p_activate_task = 0.0;// 1.0*this->dt; // how likely is it that I will activate a task each second? *dt accounts per iters per second
	this->min_task_time = 1000.0; // shortest time to complete a task
	this->max_task_time = 6000.0; // longest time to complete a task
	this->min_task_work = 1.0;
	this->max_task_work = 1.0;
	this->min_task_reward = 100.0;
	this->max_task_reward = 500.0;
	// agent stuff
	this->min_travel_vel = 2.3; // 5 - slowest travel speed
	this->max_travel_vel = 2.7; // 25 - fastest travel speed
	this->min_agent_work = 100.0; // min amount of work an agent does per second
	this->max_agent_work = 100.0; // max amount of work an agent does per second
	// agent starting locations
	this->starting_locs.push_back(cv::Point2d(-15,-15));
	this->starting_locs.push_back(cv::Point2d(15,15));
	this->starting_locs.push_back(cv::Point2d(-15,15));
	this->starting_locs.push_back(cv::Point2d(15,-15));
	this->starting_locs.push_back(cv::Point2d(0,15));
	this->starting_locs.push_back(cv::Point2d(15,0));
	this->starting_locs.push_back(cv::Point2d(0,-15));
	this->starting_locs.push_back(cv::Point2d(-15,0));
	// reset randomization
	srand(this->rand_seed);

	if(this->test_environment_img.empty()){
		this->make_obs_mat(); // create random obstacles
	}
	else{
		this->seed_obs_mat(); // seed into cells satelite information
	}
	ROS_INFO("DMCTS_world_node::   Word::World(): mat size: %i, %i (cells)", this->Obs_Mat.cols, this->Obs_Mat.rows);
	cv::Mat s = cv::Mat::zeros(this->Obs_Mat.size(), CV_8UC1);
	for(int i=0; i<this->inflation_iters; i++){
		cv::blur(this->Obs_Mat,s,cv::Size(5,5));
		cv::max(this->Obs_Mat,s,this->Obs_Mat);
	}
	// reset randomization
	srand(this->rand_seed);
	// initialize map, tasks, and agents
	this->initialize_nodes_and_tasks();
	// reset randomization
	srand(this->rand_seed);
	this->initialize_PRM();
	// initialize agents
	// reset randomization
	srand(this->rand_seed);
	this->initialize_agents(nHandle);
	this->initialized = true;

	if(this->show_display){
		this->display_world(100);
	}
}

double World::get_task_reward_at_time(const int &task_index, const double &time){
	return this->nodes[task_index]->get_reward_at_time(time);
}

void World::seed_obs_mat(){

	this->Obs_Mat = cv::Mat::zeros(cv::Size(int(this->map_width_meters), int(this->map_height_meters)), CV_8UC1);
	this->Env_Mat = cv::Mat::zeros(cv::Size(int(this->map_width_meters), int(this->map_height_meters)), CV_8UC3);

	cv::Mat temp_obs = cv::imread(this->test_obstacle_img, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat temp_env = cv::imread(this->test_environment_img, CV_LOAD_IMAGE_COLOR);

	//cv::namedWindow("DMCTS_World::World::seed_obs_mat:Obstacles", cv::WINDOW_NORMAL);
	//cv::imshow("DMCTS_World::World::seed_obs_mat:Obstacles", temp_env);
	//cv::waitKey(0);

	if(!temp_obs.data || !temp_env.data){
		ROS_ERROR("World::seed_img::Could NOT load img");
		return;
	}

	cv::resize(temp_obs, this->Obs_Mat, this->Obs_Mat.size());
	cv::resize(temp_env, this->Env_Mat, this->Env_Mat.size());

	//cv::namedWindow("DMCTS_World::World::seed_obs_mat:Obstacles", cv::WINDOW_NORMAL);
	//cv::imshow("DMCTS_World::World::seed_obs_mat:Obstacles", this->Env_Mat);
	//cv::waitKey(0);
}

void World::make_obs_mat(){
	this->map_width_meters = 100.0;
	this->map_height_meters = 100.0;
	this->Obs_Mat = cv::Mat::zeros(cv::Size(int(this->map_width_meters), int(this->map_height_meters)), CV_8UC1);
	this->Env_Mat = cv::Mat::zeros(cv::Size(int(this->map_width_meters), int(this->map_height_meters)), CV_8UC3);
	
	this->obstacles.clear();
	//ROS_INFO("DMCTS_World::World::make_obs_mat: making obstacles");
	while(this->obstacles.size() < this->n_obstacles){
		//ROS_INFO("making obstacle");
		// create a potnetial obstacle
		double rr = rand_double_in_range(1,10);
		double xx = rand_double_in_range(-this->map_width_meters/2.1,this->map_width_meters/2.1);
		double yy = rand_double_in_range(-this->map_height_meters/2.1,this->map_height_meters/2.1);
		//ROS_INFO("obs: %.1f, %.1f, r =  %.1f", xx, yy, rr);
		// check if any starting locations are in an obstacle
		bool flag = true;
		for(size_t s=0; s<this->starting_locs.size(); s++){
			double d = sqrt(pow(xx-this->starting_locs[s].x,2) + pow(yy-this->starting_locs[s].y,2));
			//ROS_INFO("starting_locs: %.1f, %.1f, d = %.1f", this->starting_locs[s].x+this->map_width_meters/2, this->starting_locs[s].y+this->map_height_meters/2, d);
			if(rr+2 >= d ){
				// starting loc is in obstacle
				flag = false;
				break;
			}
		}

		if(flag){
			for(size_t s=0; s<this->obstacles.size(); s++){
				double d = sqrt(pow(xx-this->obstacles[s][0],2) + pow(yy-this->obstacles[s][1],2));
				if(rr + this->obstacles[s][2]+1 >= d){
					// obstacle is in obstacle so don't make
					flag = false;
					break;
				}
			}			
		}
		if(flag){
			std::vector<double> temp = {xx,yy,rr};
			this->obstacles.push_back(temp);
		}
	}

	for(size_t i=0; i<this->obstacles.size(); i++){
		cv::circle(this->Obs_Mat, cv::Point((this->obstacles[i][0]+this->map_width_meters/2), (this->obstacles[i][1]+this->map_height_meters/2)), this->obstacles[i][2], cv::Scalar(255), -1);
		cv::circle(this->Env_Mat, cv::Point((this->obstacles[i][0]+this->map_width_meters/2), (this->obstacles[i][1]+this->map_height_meters/2)), this->obstacles[i][2], cv::Scalar(255,0,0), -1);
	}

	//cv::namedWindow("DMCTS_World::World::make_obs_mat:Obstacles", cv::WINDOW_NORMAL);
	//cv::imshow("DMCTS_World::World::make_obs_mat:Obstacles", this->Obs_Mat);
	//cv::waitKey(0);
}

void World::plot_timer_callback(const ros::TimerEvent &e){
	this->display_world(1);
}

void World::reset_task_status_list(){
	this->task_status_list.clear();
	for(int i=0; i<this->n_nodes; i++){
		this->task_status_list.push_back(this->nodes[i]->is_active());
	}
}

bool World::are_nbrs(const int &t1, const int &t2) {
	if (t1 >= 0 && t1 < this->n_nodes) {
		if (this->nodes[t1]->is_nbr(t2)) {
			return true;
		}
	}
	return false;
}

void World::generate_tasks() {
	// generate with certain probability each time step
	if (true) {
		double r = this->rand_double_in_range(0.0, 1.0);
		if (r < this->p_activate_task) {
			bool flag = true;
			int iter = 0;
			while (flag && iter < 100) {
				iter++;
				int c_task = rand() % this->n_nodes;
				if (!this->nodes[c_task]->is_active()) {
					this->nodes[c_task]->activate(this);
					flag = false;
				}
			}
		}
	}

	for (int n = 0; n < this->n_nodes; n++) {
		this->task_status_list[n] = this->nodes[n]->is_active();
	}
}

void World::activate_task(const int &ti){
	this->nodes[ti]->activate(this);
	this->task_status_list[ti] = true;
}

void World::deactivate_task(const int &ti){
	this->nodes[ti]->deactivate();
	this->task_status_list[ti] = false;
}

int World::get_n_active_tasks(){
	int cntr = 0;
	for(int i=0; i<this->n_nodes; i++){
		if(this->nodes[i]->is_active()){
			cntr++;
			this->task_status_list[i] = true;
		}
		else{
			this->task_status_list[i] = false;
		}
	}
	return cntr;
}

bool World::get_task_completion_time(const int &ai, const int &ti, double &time) {
	if (this->valid_agent(ai) && this->valid_node(ti)) {
		time = this->nodes[ti]->get_time_to_complete(this->agents[ai], this);
		return true;
	}
	else {
		return false;
	}
}

bool World::valid_agent(const int a) {
	if (a < this->n_agents && a > -1) {
		return true;
	}
	else {
		return false;
	}
}

bool World::get_travel_cost(const int &s, const int &g, const bool &pay_obstacle_cost, double &cost) {
	if (this->valid_node(s) && this->valid_node(g)) {
		std::vector<int> path;
		bool need_path = false;
		if (this->a_star(s, g, pay_obstacle_cost, need_path, path, cost)) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}

bool World::get_travel_time(const int &s, const int &g, const double &step_dist, const bool &pay_obstacle_cost, double &time) {
	if (this->valid_node(s) && this->valid_node(g)) {
		double dist;
		std::vector<int> path;
		bool need_path = false;
		if (this->a_star(s, g, pay_obstacle_cost, need_path, path, dist)) {
			time = dist / step_dist;
			return true;
		}
		else {
			return false;
		}
	
	}
	else {
		return false;
	}
}

bool World::valid_node(const int &n) {
	if (n < this->n_nodes && n > -1) {
		return true;
	}
	else {
		return false;
	}
}

bool World::get_task_status(const int &task_index){
	return this->nodes[task_index]->is_active(); 
}

void World::iterate_my_agent() {
	this->agents[this->my_agent_index]->act();
}

void World::get_task_status_list(std::vector<bool> &task_status_list, std::vector<int> &task_set) {
	task_set.clear();
	task_status_list.clear();

	task_status_list = this->task_status_list;
	for (size_t i = 0; i < this->task_status_list.size(); i++) {
		if (this->task_status_list[i]) {
			task_set.push_back(int(i));
		}
	}
}

void World::get_claims_after(const int &task_num, const double &query_time, const int &agent_index, std::vector<double> probs, std::vector<double> times){
	probs.clear();
	times.clear();
	for (int a = 0; a < this->n_agents; a++) {
		if (a != agent_index) { // ignore the specified agent
			Agent_Coordinator* coord = this->agents[a]->get_coordinator(); // get coordinator for readability
			coord->get_prob_actions()[task_num]->get_claims_after(query_time, probs, times); // add times and probs for other agent to list
		}
	}
}

double World::get_team_probability_at_time_except(const double &time, const int &task, const int &except_agent) {
	double p_task_I_time = 0.0;
	for (int a = 0; a < this->n_agents; a++) {
		if (a != except_agent) { // ignore the specified agent
			Agent_Coordinator* coord = this->agents[a]->get_coordinator(); // get coordinator for readability
			double p_t = coord->get_prob_actions()[task]->get_probability_at_time(time); // get probable actions of agent a
			//ROS_ERROR("World::get_team_probability_at_time_except: a[%i] has p=%0.2f at t=%0.2f for task=%i", a, p_t, time, task);
			if (p_t > 0) {
				p_task_I_time = coord->get_prob_actions()[task]->probability_update_inclusive(p_task_I_time, p_t); // add to cumulative actions of team
			}
		}
	}
	return p_task_I_time;
}

void World::display_world(const int &ms) {

	if (!this->show_display){
		return;
	}
	if(!this->initialized) {
		ROS_WARN("World::display_world: Agent %i world not initialized", this->my_agent_index);
		return;
	}

	cv::Scalar red(0.0, 0.0, 255.0);
	cv::Scalar blue(255.0, 0.0, 0.0);
	cv::Scalar green(0.0, 255.0, 0.0);
	cv::Scalar white(255.0, 255.0, 255.0);
	cv::Scalar orange(69.0, 100.0, 255.0);
	cv::Scalar black(0.0, 0.0, 0.0);
	cv::Scalar gray(127.0, 127.0, 127.0);

	double des_x, des_y, scale;

	if(this->map_width_meters > this->map_height_meters){
		des_x = 1000.0;
		scale = des_x / this->map_width_meters;
		des_y = this->map_height_meters * scale;
	}
	else{
		des_y = 1000.0;
		scale = des_y / this->map_height_meters;
		des_x = this->map_width_meters * scale;
	}

	//ROS_INFO("have des: %.2f and %.2f", des_x, des_y);

	if (this->PRM_Mat.empty()) {
		this->PRM_Mat = cv::Mat::zeros(cv::Size(des_x, des_y), CV_8UC3);
		//ROS_INFO("made it here with PRM: %i and %i", this->PRM_Mat.cols, this->PRM_Mat.rows);
		// draw obstacles
		cv::resize(this->Env_Mat, this->PRM_Mat, this->PRM_Mat.size());

		// draw PRM connections
		for (int i = 0; i < this->n_nodes; i++) {
			int index = -1;
			for(int iter=0; iter<this->nodes[i]->get_n_nbrs(); iter++){
				if (this->nodes[i]->get_nbr_i(iter, index)) {
					double obs_cost = 0.0;
					double free_dist = 0.0;
					if (this->nodes[i]->get_nbr_obstacle_cost(iter, obs_cost) && this->nodes[i]->get_nbr_distance(iter, free_dist)) {
						double max_cost = free_dist * this->obstacle_increase;
						double ratio = (obs_cost-free_dist) / max_cost;
						cv::Scalar pink(uchar(255.0*(1.0 - ratio)), uchar(255.0*(1.0 - ratio)), 255);
						cv::Point2d p1 = this->nodes[i]->get_loc();
						cv::Point2d p2 = this->nodes[index]->get_loc();
						p1.x = scale * (p1.x + this->map_width_meters/2.0);
						p1.y = scale * (p1.y + this->map_height_meters/2.0);
						p2.x = scale * (p2.x + this->map_width_meters/2.0);
						p2.y = scale * (p2.y + this->map_height_meters/2.0);
						cv::line(this->PRM_Mat, p1, p2, pink, 2);
					}
				}
			}
		}

		// draw nodes
		for (int i = 0; i < this->n_nodes; i++) {
			cv::Point2d p1 = this->nodes[i]->get_loc();
			p1.x = scale * (p1.x + this->map_width_meters/2.0);
			p1.y = scale * (p1.y + this->map_height_meters/2.0);
			cv::circle(this->PRM_Mat, p1, 5, blue, -1);
		}

		// label tasks
		for (int i = 0; i < this->n_nodes; i++) {
			double d = -5.0;
			cv::Point2d p1 = this->nodes[i]->get_loc();
			p1.x = scale * (p1.x + this->map_width_meters/2);
			p1.y = scale * (p1.y + this->map_height_meters/2);
			cv::Point2d tl = cv::Point2d(p1.x - d, p1.y + d);
			char text[10];
			sprintf(text, "%i", i);
			if (this->nodes[i]->is_active()) {
				cv::putText(this->PRM_Mat, text, tl, CV_FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255), 3);
			}
		}
	}

	cv::Mat temp = this->PRM_Mat.clone();
	cv::Mat map = cv::Mat::zeros(cv::Size(int(des_x), int(des_y) + 100), CV_8UC3);
	temp.copyTo(map(cv::Rect(0, 0, temp.cols, temp.rows)));
	
	// draw active tasks
	for (int i = 0; i < this->n_nodes; i++) {
		if (this->nodes[i]->is_active()) {
			double d = 15.0;
			cv::Point2d p1 = this->nodes[i]->get_loc();
			p1.x = scale * (p1.x + this->map_width_meters/2);
			p1.y = scale * (p1.y + this->map_height_meters/2);
			cv::Point2d tl = cv::Point2d(p1.x - d, p1.y + d);
			cv::Point2d br = cv::Point2d(p1.x + d, p1.y - d);
			cv::rectangle(map, cv::Rect(tl, br), this->nodes[i]->get_color(), -1);
		}
	}

	// draw agents
	for (int i = 0; i < this->n_agents; i++) {
		// draw their location
		if(i == this->my_agent_index){
			cv::Point2d p1(this->agents[i]->get_pose()->get_x(), this->agents[i]->get_pose()->get_y());
			p1.x = scale * (p1.x + this->map_width_meters/2); p1.y = scale * (p1.y + this->map_height_meters/2);
			if(p1.x >= 0.0 && p1.x <= des_x && p1.y >= 0.0 && p1.y <= des_y){
				cv::circle(map, p1, 8, this->agents[i]->get_color(), -1);

				// draw line to their edge x
				if(this->agents[i]->get_edge_x() >= 0 && this->agents[i]->get_edge_x() <= this->n_nodes){
					cv::Point2d p2 = this->nodes[this->agents[i]->get_edge_x()]->get_loc();
					p2.x = scale * (p2.x + this->map_width_meters/2);
					p2.y = scale * (p2.y + this->map_height_meters/2);
					cv::line(map, p1, p2, this->agents[i]->get_color(), 2);
				}

				// draw line to their edge y
				if(this->agents[i]->get_edge_y() >= 0 && this->agents[i]->get_edge_y() <= this->n_nodes){
					cv::Point2d p3 = this->nodes[this->agents[i]->get_edge_y()]->get_loc();
					p3.x = scale * (p3.x + this->map_width_meters/2);
					p3.y = scale * (p3.y + this->map_height_meters/2);
					cv::line(map, p1, p3, this->agents[i]->get_color(), 2);
				}
		
				char text[10];
				sprintf(text, "%i", i);
				cv::putText(map, text, p1, CV_FONT_HERSHEY_COMPLEX, 1.0, this->agents[i]->get_color(), 3);
			}

			std::vector<int> a_path = this->agents[i]->get_path();
			if(a_path.size() > 0){
				cv::Point p_loc = this->nodes[this->agents[i]->get_edge_y()]->get_loc();
				p_loc.x = scale * (p_loc.x + this->map_width_meters/2); p_loc.y = scale * (p_loc.y + this->map_height_meters/2);
				for(size_t j=1; j<a_path.size(); j++){
					cv::Point p_cur = this->nodes[a_path[j]]->get_loc();
					p_cur.x = scale * (p_cur.x + this->map_width_meters/2);
					p_cur.y = scale * (p_cur.y + this->map_height_meters/2);
					cv::Point p1(p_cur.x - 10*i, p_cur.y - 10*i);
					cv::Point p2(p_loc.x - 10*i, p_loc.y - 10*i);

					cv::arrowedLine(map, p2, p1, this->agents[i]->get_color(), 4);
					p_loc = p_cur;
				}
			}
		}
	}
	
	cv::putText(map, this->task_selection_method, cv::Point2d(40.0, des_y + 30.0), CV_FONT_HERSHEY_COMPLEX, 1.0, white, 3);
	char time[200];
	sprintf(time, "Time: %.2f of %.2f", this->c_time, this->end_time);
	cv::putText(map, time, cv::Point2d(30.0, des_y + 80.0), CV_FONT_HERSHEY_COMPLEX, 1.0, white, 3);
	char map_name[200];
	sprintf(map_name, "Map Name: %i", this->rand_seed);

	cv::putText(map, map_name, cv::Point2d(500.0, des_y + 80), CV_FONT_HERSHEY_COMPLEX, 1.0, white, 3);
	
	double current_plot_time = 1000.0 * clock() / double(CLOCKS_PER_SEC);
	cv::namedWindow("Map World", cv::WINDOW_NORMAL);
	cv::imshow("Map World", map);
	if (ms == 0) {
		cv::waitKey(0);
	}
	else if (double(ms) - (this->last_plot_time - current_plot_time) < 0.0) {
		cv::waitKey(100);//ms - int(floor(this->last_plot_time - current_plot_time)));
	}
	else {
		cv::waitKey(100);
	}

	this->last_plot_time = 1000.0 * clock() / double(CLOCKS_PER_SEC);
}


void World::initialize_agents(ros::NodeHandle nHandle) {
	//ROS_INFO("World::initialize_agents: my agent index is: %i", this->my_agent_index);

	std::vector<double> agent_travel_vels;
	//fs["agent_travel_vels"] >> agent_travel_vels;
	std::vector<bool> agent_obstacle_costs;
	//fs["agent_obstacle_costs"] >> agent_obstacle_costs;
	std::vector<double> agent_work_radii;
	//fs["agent_work_radii"] >> agent_work_radii;
	std::vector<int> agent_types;
	//fs["agent_types"] >> agent_types;
	for(int i=0; i<this->n_agents; i++){
		agent_types.push_back(0);
	}

	std::vector<cv::Scalar> agent_colors;
	for (int i = 0; i < this->n_agent_types; i++) {
		agent_travel_vels.push_back(2.0);
		agent_obstacle_costs.push_back(this->pay_obstacle_cost);
		agent_work_radii.push_back(0.1);
		if(this->my_agent_index == i){
			double r = 255.0;
			double b = 0.0;
			double g = 0.0;

			cv::Scalar color(b, g, r);
			agent_colors.push_back(color);	
		}
		else{
			double r = 127.0;
			double b = 127.0;
			double g = 127.0;

			cv::Scalar color(b, g, r);
			agent_colors.push_back(color);
		}
	}

	bool actual_agent = false;
	for (int i = 0; i < this->n_agents; i++) {
		int tp = agent_types[i];
		if(i == this->my_agent_index){
			actual_agent = true;
		}
		Agent* a = new Agent(nHandle, i, this->my_agent_type, this->agent_cruising_speed, agent_colors[tp], agent_obstacle_costs[tp], agent_work_radii[tp], actual_agent, this, this->desired_alt);
		this->agents.push_back(a);
		actual_agent = false;
	}
}

void World::initialize_PRM() {
	// get travel distance between all nodes
	this->travel_distances = cv::Mat(this->n_nodes, this->n_nodes, CV_32F, -1);
	this->obstacle_distances = cv::Mat(this->n_nodes, this->n_nodes, CV_32F, -1);


	// connect all nodes within radius
	this->task_status_list.clear();
	for (int i = 0; i < this->n_nodes; i++) {
		// all tasks are started off
		this->task_status_list.push_back(false);
		for (int j = i+1; j < this->n_nodes; j++) {
			double d;
			if (this->dist_between_nodes(i, j, d)) {
				if (d < this->k_connection_radius && rand() < this->p_connect) { // am I close enough?
	
					// set normal nbr and travel
					double obs_cost = this->find_obstacle_costs(i,j,d);
					this->nodes[i]->add_nbr(j, d, obs_cost);
					this->nodes[j]->add_nbr(i, d, obs_cost);
					this->travel_distances.at<float>(i,j) = d;
					this->obstacle_distances.at<float>(i,j) = obs_cost;
					this->travel_distances.at<float>(j,i) = d;
					this->obstacle_distances.at<float>(j,i) = obs_cost;
				}
			}
		}

		// if not enough connections were done, add some more connections until min is reached
		this->k_map_connections = std::min(this->k_map_connections, this->n_nodes-1);
		while(this->nodes[i]->get_n_nbrs() < this->k_map_connections){
			double min_dist = double(INFINITY);
			int mindex = -1;
			
			for (int j = 0; j < this->n_nodes; j++) {
				if (i != j) {
					bool in_set = false;
					for (int k = 0; k < this->nodes[i]->get_n_nbrs(); k++) {
						int n_i = 0;
						if (this->nodes[i]->get_nbr_i(k,n_i)){
							if (n_i == j) {
								in_set = true;
								break;
							}
						}
					}
					if (!in_set) {
						double d;
						if (this->dist_between_nodes(i, j, d)) {
							if (d < min_dist) {
								min_dist = d;
								mindex = j;
							}
						}
					}
				}
			}
			//cv::namedWindow("obstacles", cv::WINDOW_NORMAL);
			//cv::imshow("obstacles", this->Obs_Mat);
			//cv::waitKey(100);

			double obs_cost = this->find_obstacle_costs(i,mindex, min_dist);
			this->nodes[i]->add_nbr(mindex, min_dist, obs_cost);
			this->nodes[mindex]->add_nbr(i, min_dist, obs_cost);
			this->travel_distances.at<float>(i,mindex) = min_dist;
			this->obstacle_distances.at<float>(i,mindex) = obs_cost;
			this->travel_distances.at<float>(mindex,i) = min_dist;
			this->obstacle_distances.at<float>(mindex,i) = obs_cost;
			//ROS_INFO("%i -> %i: %0.1f", i, mindex, min_dist);
		}
	}

	/*
	ROS_ERROR("travel_distances");
	for(int i=0; i<this->n_nodes; i++){
		for(int j=0; j<this->n_nodes; j++){
			std::cout << this->obstacle_distances.at<float>(i,j) << ",";
		}
		std::cout << std::endl;
	}
	*/

	for(int i=0; i<this->n_nodes; i++){
		for(int j=i; j<this->n_nodes; j++){
			if(this->travel_distances.at<float>(i,j) == -1){
				double d = 0.0;
				std::vector<int> path;
				if(a_star(i, j, false, false, path, d)){
					this->travel_distances.at<float>(i,j) = float(d);
					this->travel_distances.at<float>(j,i) = float(d);
				}
				else{
					this->travel_distances.at<float>(i,j) = INFINITY;
					this->travel_distances.at<float>(j,i) = INFINITY;
				}
				if(a_star(i, j, true, false, path, d)){
					this->obstacle_distances.at<float>(i,j) = float(d);
					this->obstacle_distances.at<float>(j,i) = float(d);
				}
				else{
					this->obstacle_distances.at<float>(i,j) = INFINITY;
					this->obstacle_distances.at<float>(j,i) = INFINITY;
				}
			}
		}
	}

	/*
	ROS_ERROR("travel_distances prime");
	for(int i=0; i<this->n_nodes; i++){
		for(int j=0; j<this->n_nodes; j++){
			std::cout << this->obstacle_distances.at<float>(i,j) << ",";
		}
		std::cout << std::endl;
	}
	*/
}

double World::find_obstacle_costs(const int &i, const int &j, const double &free_dist){
	// get node[i]'s location on the img
	cv::Point me = this->nodes[i]->get_loc();
	me.x += this->map_width_meters/2;
	me.y += this->map_height_meters/2;
	// get their location
	cv::Point np = this->nodes[j]->get_loc();
	np.x += this->map_width_meters/2;
	np.y += this->map_height_meters/2;
	// get a line iterator from me to them
	cv::LineIterator lit(this->Obs_Mat, me, np);
	double val_sum = 0.0;
	// count every obstacles cell between me and them!
	for (int i = 0; i < lit.count; i++, ++lit) {
		// count along line
		val_sum += double(this->Obs_Mat.at<uchar>(lit.pos()))/255.0;
	}
	double mean_val = val_sum / double(lit.count);
	double obs_cost = free_dist + (free_dist*mean_val*this->obstacle_increase);
	return obs_cost;
}

double World::rand_double_in_range(const double &min, const double &max) {
	// get random double between min and max
	return (max - min) * double(rand()) / double(RAND_MAX) + min;
}

double World::dist2d(const double &x1, const double &x2, const double &y1, const double &y2) {
	// get distance between 1 and 2
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

bool World::get_edge_cost(const int &n1, const int &n2, const bool &pay_obstacle_cost, double &cost) {
	//std::cerr << "World::get_edge_cost: with edge: " << n1 << " -> " << n2 << std::endl;
	if (n1 >= 0 && n1 < this->n_nodes) {
		if (n1 == n2) {
			cost = 0.0;
			return true;
		}
		int nbr_index = -1;
		for (int ni = 0; ni < this->nodes[n1]->get_n_nbrs(); ni++) {
			if (this->nodes[n1]->get_nbr_i(ni, nbr_index)) {
				if (nbr_index == n2) {
					this->nodes[n1]->get_nbr_travel_cost(ni, pay_obstacle_cost, cost);
					return true;
				}
			}
		}
	}
	return false;
}

bool World::dist_between_nodes(const int &n1, const int &n2, double &d) {
	// am I a node?
	if (n1 >= 0 && n1 < this->n_nodes && n2 >= 0 && n2 < this->n_nodes) {
		// get distance between nodes
		d = sqrt(pow(this->nodes[n1]->get_x() - this->nodes[n2]->get_x(), 2) + pow(this->nodes[n1]->get_y() - this->nodes[n2]->get_y(), 2));
		return true;
	}
	else {
		return false;
	}
}

void World::initialize_nodes_and_tasks() {
	this->min_task_times.clear();
	this->max_task_times.clear();
	this->min_task_rewards.clear();
	this->max_task_rewards.clear();
	this->min_task_works.clear();
	this->max_task_works.clear();
	std::vector<cv::Scalar> task_colors;
	std::vector<std::vector<double> > task_work_by_agent;
	for (int t = 0; t < this->n_task_types; t++) {

		double r = rand_double_in_range(0.0, 255.0);
		double b = rand_double_in_range(0.0, 255.0);
		double g = rand_double_in_range(0.0, 255.0);

		cv::Scalar color(b, g, r);
		task_colors.push_back(color);

		double t0 = this->rand_double_in_range(this->min_task_time, this->max_task_time);
		double t1 = this->rand_double_in_range(this->min_task_time, this->max_task_time);
		this->min_task_times.push_back(std::min(t0,t1));
		this->max_task_times.push_back(std::max(t0,t1));

		t0 = this->rand_double_in_range(this->min_task_reward, this->max_task_reward);
		t1 = this->rand_double_in_range(this->min_task_reward, this->max_task_reward);
		this->min_task_rewards.push_back(std::min(t0,t1));
		this->max_task_rewards.push_back(std::max(t0,t1));

		t0 = this->rand_double_in_range(this->min_task_work, this->max_task_work);
		t1 = this->rand_double_in_range(this->min_task_work, this->max_task_work);
		this->min_task_works.push_back(std::min(t0,t1));
		this->max_task_works.push_back(std::max(t0,t1));

		// set task times
		std::vector<double> at;
		if(this->hardware_trial){ // for search and rescue
			if(t==0){
				// open field task
				at.push_back(1000000.0);
				at.push_back(1000000.0);

			}
			else{
				// obstacle task
				at.push_back(1000000.0); // low flying agents can see
				at.push_back(0.0); // high flying agents can not
			}
		}
		else{
			// random trials
			for (int a = 0; a < this->n_agent_types; a++) {
				if (this->rand_double_in_range(0.0, 1.0) < this->p_impossible_task) {
					at.push_back(double(INFINITY));
				}
				else {
					at.push_back(this->rand_double_in_range(this->min_agent_work, this->max_agent_work));
				}
			}
		}
		task_work_by_agent.push_back(at);
	}

	double x,y;
	for(size_t i=0; i<this->starting_locs.size(); i++){
		int task_type = rand() % n_task_types;
		Map_Node* n = new Map_Node(this->starting_locs[i].x, this->starting_locs[i].y, i, this->p_task_initially_active, task_type, task_work_by_agent[task_type], task_colors[task_type], this->flat_tasks, this);
		this->nodes.push_back(n);
	}

	int obs_threshold = 50;
	if(this->hardware_trial){
		obs_threshold = 255;
	}

	for(int i=int(this->nodes.size()); i < this->n_nodes; i++) {
		bool flag = true;

		while(flag){
			x = this->rand_double_in_range(-this->map_width_meters/2.1, this->map_width_meters/2.1);
			y = this->rand_double_in_range(-this->map_height_meters/2.1, this->map_height_meters/2.1);

			if(this->Obs_Mat.at<uchar>(cv::Point(x + this->map_width_meters/2,y+this->map_height_meters/2)) <= obs_threshold){
				flag = false;
			}
		}

		int task_type = rand() % n_task_types;
		if(this->hardware_trial){
			if(this->Obs_Mat.at<uchar>(cv::Point(x + this->map_width_meters/2,y+this->map_height_meters/2)) <= 50){
				task_type = 0; // open field task
			}
			else{
				task_type = 1; // obstacle task
			}
		}
		Map_Node* n = new Map_Node(x, y, i, this->p_task_initially_active, task_type, task_work_by_agent[task_type], task_colors[task_type], this->flat_tasks, this);
		this->nodes.push_back(n);
	}
}

World::~World(){
	for (size_t i = 0; i < this->nodes.size(); i++) {
		delete this->nodes[i];
	}
	this->nodes.clear();

	for (size_t i = 0; i < this->agents.size(); i++) {
		delete this->agents[i];
	}
	this->agents.clear();
}

bool World::a_star(const int &start, const int &goal, const bool &pay_obstacle_cost, const bool &need_path, std::vector<int> &path, double &length) {

	if (start < 0 || start >= this->n_nodes) {
		ROS_ERROR("World::a_star:: start off graph");
		return false;
	}
	if (goal < 0 || goal >= this->n_nodes) {
		ROS_ERROR("World::a_star:: goal off graph");
		return false;
	}

	if (!need_path){
		// Don't need path, check the obstacle and travel distance mats
		if(pay_obstacle_cost && this->obstacle_distances.at<float>(start,goal) > 0){
			length = this->obstacle_distances.at<float>(start,goal);
			return true;
		}
		if(!pay_obstacle_cost && this->travel_distances.at<float>(start,goal) > 0){
			length = this->travel_distances.at<float>(start,goal);
			return true;
		}
	}

	// garbage variables
	int trash_i = -1;
	double trash_d = -1.0;

	// The set of nodes already evaluated
	std::vector<int> closed_set;

	// The set of currently discovered nodes that are not evaluated yet.
	// Initially, only the start node is known.
	std::vector<int> open_set;
	open_set.push_back(start);

	// For each node, which node it can most efficiently be reached from.
	// If a node can be reached from many nodes, cameFrom will eventually contain the
	// most efficient previous step.
	std::vector<int> came_from(this->n_nodes, -1);

	// For each node, the cost of getting from the start node to that node.
	std::vector<double> gScore(this->n_nodes, double(INFINITY));

	// The cost of going from start to start is zero.
	gScore[start] = 0.0;

	// For each node, the total cost of getting from the start node to the goal
	// by passing by that node. That value is partly known, partly heuristic.
	std::vector<double> fScore(this->n_nodes, double(INFINITY));
	

	// For the first node, that value is completely heuristic.
	this->dist_between_nodes(start, goal, fScore[start]);

	while (open_set.size() > 0) {
		// the node in openSet having the lowest fScore[] value
		int current = -1;
		if (!get_mindex(fScore, current, trash_d)) {
			return false;
		}
		if (current == goal) {
			path.clear();
			length = gScore[current];
			path.push_back(current);
			while (current != start) {
				current = came_from[current];
				path.push_back(current);
			}
			return true;
		}

		int index = 0;
		if (this->get_index(open_set, current, index)) {
			fScore[current] = double(INFINITY);
			open_set.erase(open_set.begin() + index);
		}
		closed_set.push_back(current);

		int n_nbrs = this->nodes[current]->get_n_nbrs();
		for (int ni = 0; ni < n_nbrs; ni++) {
			int neighbor = -1;
			if (this->nodes[current]->get_nbr_i(ni, neighbor)) {
				if (this->get_index(closed_set, neighbor, trash_i)) { // returns false if not in set
					continue;	// Ignore the neighbor which is already evaluated.
				}

				if (!this->get_index(open_set, neighbor, trash_i)) {// Discover a new node
					open_set.push_back(neighbor);
				}

				// The distance from start to a neighbor
				//if (this->dist_between_nodes(current, neighbor, trash_d)) {
				if(this->nodes[current]->get_nbr_travel_cost(ni, pay_obstacle_cost, trash_d)){
					double tentative_gScore = gScore[current] + trash_d;
					if (tentative_gScore >= gScore[neighbor]) {
						continue;		// This is not a better path.
					}

					// This path is the best until now. Record it!
					came_from[neighbor] = current;
					gScore[neighbor] = tentative_gScore;
					if (this->dist_between_nodes(neighbor, goal, trash_d)) {
						fScore[neighbor] = gScore[neighbor] + trash_d;
					}
				}
			}
		}
	}
	return false;
}

bool World::get_mindex(const std::vector<double> &vals, int &mindex, double &minval) {
	minval = double(INFINITY);
	mindex = -1;

	for (size_t i = 0; i < vals.size(); i++) {
		if (vals[i] < minval) {
			minval = vals[i];
			mindex = int(i);
		}
	}
	if (mindex == -1) {
		return false;
	}
	else {
		return true;
	}
}

bool World::get_index(const std::vector<int> &vals, const int &key, int &index) {
	for (index = 0; index < int(vals.size()); index++) {
		if (vals[index] == key) {
			return true;
		}
	}
	return false;
}

double World::get_global_distance(const double &lata, const double &lona, const double &latb, const double &lonb){
	double R = 6378136.6; // radius of the earth in meters

	double lat1 = this->to_radians(lata);
	double lon1 = this->to_radians(lona);
	double lat2 = this->to_radians(latb);
	double lon2 = this->to_radians(lonb);

	double dlon = lon2 - lon1;
	double dlat = lat2 - lat1;

	double a = pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dlon / 2), 2);
	double c = 2 * atan2(sqrt(a), sqrt(1 - a));

	double distance = R * c; // in meters
	return distance;
}

double World::to_radians(const double &deg){
	return deg*0.017453292519943;
}



/**************** RETIRED FUNCTIONS *************************************
void World::create_random_obstacles() {
this->obstacle_mat = cv::Mat::zeros(int(this->map_width), int(this->map_height), CV_8UC3);

for (int i = 0; i < this->n_obstacles; i++) {
int r = (rand() %  (this->max_obstacle_radius - this->min_obstacles_radius)) + this->min_obstacles_radius;
cv::Point c;
c.x = rand() % int(this->map_width);
c.y = rand() % int(this->map_height);
cv::circle(this->obstacle_mat, c, r, cv::Vec3b(0,0,100), -1);
}
}

void World::find_obstacle_costs() {
// go through each node and for each of their nbrs, get the obstacle cost for going through each node
for (int i = 0; i < this->n_nodes; i++) {
// get node[i]'s location on the img
cv::Point me = this->nodes[i]->get_loc();
// go through all node[i]'s nbrs
int iter = 0;
int ni = 0;
// get their node[?] index
while(this->nodes[i]->get_nbr_i(iter, ni)) {
// get their location
cv::Point np = this->nodes[ni]->get_loc();
// get a line iterator from me to them
cv::LineIterator lit(this->obstacle_mat, me, np);
double val_sum = 0.0;
// count every obstacles cell between me and them!
for (int i = 0; i < lit.count; i++, ++lit) {
// count along line
if (this->obstacle_mat.at<cv::Vec3b>(lit.pos()) == cv::Vec3b(0,0,100)) {
// hit an obstacle
val_sum++;
}
}
double mean_val = val_sum / double(lit.count);
this->nodes[i]->set_nbr_obstacle_cost(iter, mean_val);
iter++;
}
}

// draw PRM connections
for (int i = 0; i < this->n_nodes; i++) {
int index = -1;
int iter = 0;
while (this->nodes[i]->get_nbr_i(iter, index)) {
iter++; // tracks which nbr I am on
double cost = 0.0;
if (this->nodes[i]->get_nbr_obstacle_cost(iter, cost)) {
cv::line(this->obstacle_mat, this->nodes[i]->get_loc(), this->nodes[index]->get_loc(), cv::Scalar(0,0,0), 5);
cv::Vec3b pink(255.0*(1 - cost), 255.0*(1 - cost), 255);
cv::line(this->obstacle_mat, this->nodes[i]->get_loc(), this->nodes[index]->get_loc(), pink, 2);
}
}
}

// draw nodes
for (int i = 0; i < this->n_nodes; i++) {
cv::circle(this->obstacle_mat, this->nodes[i]->get_loc(), 5, cv::Scalar(255,255,255), -1);
}

// label tasks
for (int i = 0; i < this->n_nodes; i++) {
double d = -5.0;
cv::Point2d l = this->nodes[i]->get_loc();
cv::Point2d tl = cv::Point2d(l.x - d, l.y + d);
char text[10];
sprintf_s(text, "%i", i);
cv::putText(this->obstacle_mat, text, tl, CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 3);
}

}
*/
