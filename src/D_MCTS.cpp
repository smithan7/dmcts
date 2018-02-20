#include "D_MCTS.h"
#include "World.h"
#include "Agent.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Pose.h"

#include <iostream>
#include "ros/ros.h"


D_MCTS::D_MCTS(World* world, Map_Node* task_in, Agent* agent_in, D_MCTS* parent, const int &my_kid_index, const double &parent_time_in){
	//std::cerr << "-a" << std::endl;
	this->agent = agent_in;
	this->task = task_in;
	this->task_index = task_in->get_index();
	this->world = world;
	this->last_update_time = this->world->get_c_time();
	//std::cerr << "a" << std::endl;
	if (parent) {
		this->probability = -1.0;
		this->parent = parent;
		this->completion_time = -1.0;
	}
	else {
		// Set as root if I don't have a parent
		this->probability = 1.0;
		this->parent = NULL;
		if (task_in->is_active()) {
			// this is the root! I am already here!!!
			this->completion_time = parent_time_in;
		}
		else {
			// this is the root! I am already here!!!
			this->completion_time = parent_time_in + task_in->get_time_to_complete(this->agent, world);
		}
	}

	//std::cerr << "b" << std::endl;
	this->parent_time = parent_time_in;

	// a star distance checks
	this->distance = -1.0; // how far by A*
	this->travel_time = -1.0; // when will I arrive by A*
	this->reward = -1.0; // reward * (1-p_taken)
	this->expected_reward = 0.0; // expected reward of this child

	// task stuff
	this->probability_task_available = 1.0;
	this->work_time = -1.0;
	
	// MCTS stuff
	this->branch_reward = 0.0; // my expected reward + all my best kids expected reward
	this->number_pulls = 0; // how many times have I been pulled
	this->max_rollout_depth = 3; //edit these to ensure tree can grow on simple case
	this->max_search_depth = 20; 

	// sampling stuff
	this->max_kid_index = -1; // index of gc
	this->max_kid_branch_reward = 0.0; // their reward

	//std::cerr << "c" << std::endl;
	// few useful constants
	if (this->world->get_task_selection_method() == "mcts_task_by_completion_reward") {
		this->reward_weighting = 1.0; // how important is the reward in the reward function
		this->distance_weighting = 0.01; // how important is the travel cost in the reward function
	}
	else if (this->world->get_task_selection_method() == "mcts_task_by_completion_value") {
		this->reward_weighting = 1.0; // how important is the reward in the reward function
		this->distance_weighting = 1.0; // how important is the travel cost in the reward function
	}
	else if (this->world->get_task_selection_method() == "mcts_task_by_completion_reward_impact") {
		this->reward_weighting = 1.0; // how important is the reward in the reward function
		this->distance_weighting = 0.01; // how important is the travel cost in the reward function
	}
	else if (this->world->get_task_selection_method() == "mcts_task_by_completion_reward_gradient") {
		this->reward_weighting = 1.0; // how important is the reward in the reward function
		this->distance_weighting = 0.0; // how important is the travel cost in the reward function
	}
	else{
		ROS_ERROR("D_MCTS::D_MCTS::get_task_selection_method is null");
		this->reward_weighting = 1.0;
		this->distance_weighting = 0.01;
	}
	//std::cerr << "d" << std::endl;

	this->search_type = this->world->get_mcts_search_type();
	this->alpha = 0.05; // gradient descent rate, how fast should my team 
	this->beta = 0.0141;//1.41; // 0.0141 this is what I found in Matlab tests//20.0; //1.41; // ucb = 1.41, d-ucb = 1.41, sw-ucb = 0.705
	this->epsilon = 0.05;//0.5; // ucb = 0.5, d-ucb = 0.05, sw-ucb = 0.05
	this->gamma = 0.9; // ucb = n/a~1.0, d-ucb = 0.9, sw-ucb = 0.9
	this->window_width = 5000; // how far back in my search history should I include searches
	this->sampling_probability_threshold = 0.05; // how low of probability will I continue to sample and report

	//std::cerr << "e" << std::endl;
}

bool D_MCTS::make_kids( std::vector<bool> &task_status, std::vector<int> &task_set ) {
	//ROS_WARN("D_MCTS::make_kids: in with task_set.size(): %i", int(task_set.size()));
	// Make kids
	if (this->kids.size() > 0) {
		for (size_t i = 0; i < this->kids.size(); i++) {
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
		this->kids.clear();
	}

	bool kids_made = false;
	// potentially add a kid for each active task
	for (size_t j = 0; j < task_set.size(); j++) {
		//ROS_WARN("D_MCTS::make_kids: checking kid[%i]", int(j));
		// which task am I looking at
		int ti = task_set[j];
		// if task ti needs to be completed
		if (task_status[ti]) {
			//ROS_WARN("D_MCTS::make_kids: task[%i] is active", int(j));
			D_MCTS* kiddo = new D_MCTS(this->world, this->world->get_nodes()[ti], this->agent, this, int(this->kids.size()), this->completion_time);
			kiddo->get_expected_reward();
			this->kids.push_back(kiddo);
			kids_made = true;
		}
	}
	return kids_made;
}

void D_MCTS::set_task_index(const int &ti) {
	this->task_index = ti;
	this->parent_time = -1.0;
}

D_MCTS::~D_MCTS(){}

void D_MCTS::sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in) {
	if (this->completion_time > this->world->get_end_time()) {
		return;
	}
	
	// add my task to coordinator
	coord_in->add_stop_to_my_path(this->task_index, this->completion_time, this->probability);
	if(this->kids.size() == 0){
		return;
	}
	// sample my children and assign probability
	this->find_kid_probabilities();
	// those kids who are good enough I should continue to sample
	for (size_t i = 0; i < this->kids.size(); i++) {
		//ROS_INFO("D_MCTS::sample_tree_and_advertise_task_probabilities: kid[%i] probs: %0.2f and thresh is %0.2f", int(i), this->kids[i]->get_probability(), this->sampling_probability_threshold);
		if (this->kids[i]->get_probability() > this->sampling_probability_threshold) {
			//std::cerr << "D_MCTS::sample_tree_and_advertise_task_probabilities: this->kids[]->get_task_index(): " << this->kids[i]->get_task_index() << std::endl;
			this->kids[i]->sample_tree_and_advertise_task_probabilities(coord_in);
		}
	}
}

void D_MCTS::update_probable_actions() {
	if (this->completion_time > this->world->get_end_time()) {
		return;
	}
	
	if(this->kids.size() == 0){
		return;
	}
	// sample my children and assign probability
	this->find_kid_probabilities();
	// those kids who are good enough I should continue to sample
	for (size_t i = 0; i < this->kids.size(); i++) {
		//ROS_INFO("D_MCTS::sample_tree_and_advertise_task_probabilities: kid[%i] probs: %0.2f and thresh is %0.2f", int(i), this->kids[i]->get_probability(), this->sampling_probability_threshold);
		if (this->kids[i]->get_probability() > this->sampling_probability_threshold) {
			//std::cerr << "D_MCTS::sample_tree_and_advertise_task_probabilities: this->kids[]->get_task_index(): " << this->kids[i]->get_task_index() << std::endl;
			this->kids[i]->update_probable_actions();
		}
	}
}

void D_MCTS::find_kid_probabilities() {
	if(this->kids.size() == 0){
		return;
	}

	// for all kids, assign their probability
	// Use Gradient descent to get kid probabilities
	if(this->kids[0]->get_probability() == -1.0){
		// need to initialize probabilities
		double sumR = 0.0;
		for(size_t i=0; i<this->kids.size(); i++){
			sumR += this->kids[i]->get_branch_reward();
		}
		// Set initial probs
		for(size_t i=0; i<this->kids.size(); i++){
			double p = this->probability * (this->kids[i]->get_branch_reward() / sumR);
			this->kids[i]->set_probability(p);
		}
	}
	else{
		// Probabilities are initialized, start gradient descent
		// Get max kid
		int maxK = -1;
		double maxR = -INFINITY;
		for (size_t i = 0; i < this->kids.size(); i++) {
			if(this->kids[i]->get_branch_reward() > maxR){
				maxR = this->kids[i]->get_branch_reward();
				maxK = i;
			}
		}
		double pSum = 0.0;
		//ROS_WARN("D_MCTS::find_kid_probabilities: maxK: %i (task %i) with maxR: %0.1f", maxK, this->kids[maxK]->get_task_index(), maxR);
		// Adjust probabilities
		for (size_t i=0; i<this->kids.size(); i++){
			//ROS_WARN("D_MCTS::find_kid_probabilities: p_init[%i]: %0.2f", int(i), this->kids[i]->get_probability());
			if (int(i) == maxK){
				// I am the best kid, increase my probability
				double p = this->kids[i]->get_probability() + this->kids[i]->get_alpha() * (1.0 - this->kids[i]->get_probability());
				this->kids[i]->set_probability(p);
			}
			else{
				// I am not the best kid, decrease my probability
				double p = this->kids[i]->get_probability() + this->kids[i]->get_alpha() * (0.0 - this->kids[i]->get_probability());
				this->kids[i]->set_probability(p);
			}
			//ROS_WARN("D_MCTS::find_kid_probabilities: p_final[%i]: %0.2f", int(i), this->kids[i]->get_probability());
			pSum += this->kids[i]->get_probability();
		}
		// Normalize Probabilities
		for(size_t i=0; i<this->kids.size(); i++){
			double p = this->kids[i]->get_probability() / pSum;
			this->kids[i]->set_probability( this->probability * p );
		}
	}
}

bool D_MCTS::ucb_select_kid(D_MCTS* &gc) {
	
	gc = NULL;
	double maxV = -INFINITY;
	double minV = INFINITY;
	

	for (size_t i = 0; i < this->kids.size(); i++) { // check all of my kids
		if ( this->kids[i]->get_branch_reward() > maxV) { // is their a* reward better than maxval?				
			maxV = this->kids[i]->get_branch_reward();
		}
		if (this->kids[i]->get_branch_reward() < minV){
			minV = this->kids[i]->get_branch_reward();
		}
	}
	double dV = maxV - minV;
	double maxM = -1;
	if(maxV > 0){
		for(size_t i=0; i<this->kids.size(); i++){
			double rr = (this->kids[i]->get_branch_reward() - minV) / dV;
			double ee = this->beta * sqrt(this->epsilon * log(this->number_pulls) / std::max(0.1, this->kids[i]->get_n_pulls()));
			if (rr + ee > maxM){
				maxM = rr + ee;
				gc = this->kids[i];
			}
		}	
	}
	if (maxM > 0.0) {
		return true;
	}
	else {
		return false;
	}
}

double D_MCTS::get_branch_reward() {
	if (this->kids.size() == 0) {
		this->branch_reward = this->get_expected_reward();
	}
	else {
		this->branch_reward = this->get_expected_reward() + this->max_kid_branch_reward;
	}
	return this->branch_reward;
}

double D_MCTS::get_expected_reward() {
	// DOES NOT INCLUDE KIDS! NOT BRANCH VALUE!
	bool need_path = false;

	if (this->work_time < 0.0) {
		this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	}

	// being set for the first time?
	if (this->distance < 0.0) {
		this->last_update_time = this->world->get_c_time();
		// need to set everything, then return reward
		double dist = 0.0;
		std::vector<int> path;
		if (this->world->a_star(this->task_index, this->parent->get_task_index(), this->agent->get_pay_obstacle_cost(), need_path, path, dist)) {
			this->distance = dist;
			this->travel_time = this->distance / this->agent->get_travel_vel();
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			//ROS_INFO("initialized node: %i with completion_time: %0.1f", this->task_index, this->completion_time);

			this->reward = this->task->get_reward_at_time(this->completion_time);
			//ROS_INFO("has reward %0.2f", this->reward);
			double p_taken = 0.0;
			if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
				//if(p_taken > 0.0){
				//	ROS_ERROR("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
				//}
				//else{
				//	ROS_WARN("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
				//}
				this->probability_task_available = (1 - p_taken);
				this->expected_reward *= this->probability_task_available;
				//ROS_INFO("has reward %0.2f", this->reward);
			}
			this->expected_reward = this->reward_weighting*this->reward - this->distance_weighting*this->distance;
			//ROS_INFO("has expected_reward %0.2f", this->expected_reward);
			//ros::Duration d = ros::Duration(1.0);
			//d.sleep();
		}
		else {
			this->expected_reward = -double(INFINITY);
			std::cout << "mcts::bad A* query" << std::endl;
		}

		return this->expected_reward;
	}

	// new arrival time, already know distance
	if (this->completion_time < 0.0) {
		this->last_update_time = this->world->get_c_time();
		if (this->agent->get_edge().y == this->task_index) {
			double cost = 0.0;
			if (this->world->get_edge_cost(this->agent->get_edge().x, this->agent->get_edge().y, this->agent->get_pay_obstacle_cost(), cost)) {
				this->distance = (1 - this->agent->get_edge_progress()) * cost;
				this->travel_time = this->distance / this->agent->get_travel_vel();
			}
		}
		else {
			double path_cost = 0.0;
			double current_edge_cost = 0.0;
			if (this->world->get_edge_cost(this->agent->get_edge().x, this->agent->get_edge().y, this->agent->get_pay_obstacle_cost(), current_edge_cost)) {
				if (this->world->get_travel_cost(this->agent->get_edge().y, this->task_index, this->agent->get_pay_obstacle_cost(), path_cost)) {
					this->distance = path_cost  + (1 - this->agent->get_edge_progress()) * current_edge_cost;
					this->travel_time = this->distance / this->agent->get_travel_vel();
				}
			}
		}

		this->completion_time = this->parent_time + this->travel_time + this->work_time;
		this->reward = this->task->get_reward_at_time(this->completion_time);
		double p_taken = 0.0;
		if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
			//if(p_taken > 0.0){
			//	ROS_ERROR("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
			//}
			//else{
			//	ROS_WARN("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
			//}
			this->probability_task_available = (1 - p_taken);
			this->reward *= this->probability_task_available;
		}
		this->expected_reward = this->reward_weighting*this->reward - this->distance_weighting*this->distance;
		return this->expected_reward;
	}

	// periodic updates
	if(this->world->get_c_time() - this->last_update_time > 1.0){
		this->last_update_time = this->world->get_c_time();
		this->reward = this->task->get_reward_at_time(this->completion_time);
		double p_taken = 0.0;
		if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
			//if(p_taken > 0.0){
			//	ROS_ERROR("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
			//}
			//else{
			//	ROS_WARN("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
			//}
			this->probability_task_available = (1 - p_taken);
			this->reward *= this->probability_task_available;
		}
		this->expected_reward = this->reward_weighting*this->reward - this->distance_weighting*this->distance;
		return this->expected_reward;
	}

	// next chance is that the probabilities changed because of a report
	if (this->probability_task_available < 0) {
		this->last_update_time = this->world->get_c_time();
		double p_taken = 0.0;
		this->reward = this->task->get_reward_at_time(this->completion_time);
		if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
			//if(p_taken > 0.0){
			//	ROS_ERROR("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
			//}
			//else{
			//	ROS_WARN("D_MCTS::get_expected_reward:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
			//}
			this->probability_task_available = (1 - p_taken);
			this->reward *= this->probability_task_available;
		}
		this->expected_reward = this->reward_weighting*this->reward - this->distance_weighting*this->distance;

		return this->expected_reward;
	}

	// don't need to set reward, just return
	return this->expected_reward;
}

void D_MCTS::update_kid_rewards_with_new_probabilities() {
	// need to update each kids expected reward and then get their updated branch reward
	if(this->kids.size() > 0){
		for (size_t i = 0; i < this->kids.size(); i++) {
			this->kids[i]->reset_task_availability_probability();
			this->kids[i]->get_expected_reward();
			this->kids[i]->get_branch_reward();
		}
}
}

void D_MCTS::reset_mcts_team_prob_actions(){
	this->reset_task_availability_probability();
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->reset_task_availability_probability();
	}	
}

void D_MCTS::search_from_root(std::vector<bool> &task_status, std::vector<int> &task_set) {
	//std::cerr << "D_MCTS::search_from_root: in" << std::endl;
	//std::cerr << "D_MCTS::search_from_root: on edge " << this->agent->get_edge().x << " -> " << this->agent->get_edge().y << std::endl;

	//ROS_INFO("D_MCTS::search_from_root: in on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	//make sure work time is set
	if (this->work_time < 0.0) {
		this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	}

	// update my reward and probability i am taken
	this->update_kid_rewards_with_new_probabilities();

	// make sure completion and travel time are correct
	double path_cost = 0.0;
	double current_edge_cost = 0.0;
	//ROS_INFO("D_MCTS::search_from_root: into updating get_edge_cost with edge %i and %i", this->agent->get_edge().x, this->agent->get_edge().y);
	if (this->world->get_edge_cost(this->agent->get_edge().x, this->agent->get_edge().y, this->agent->get_pay_obstacle_cost(), current_edge_cost)) {
		if (this->world->get_travel_cost(this->agent->get_edge().y, this->task_index, this->agent->get_pay_obstacle_cost(), path_cost)) {
			//ROS_INFO("D_MCTS::search_from_root: got edge_cost %0.2f and edge_progress %0.2f for edge %i -> %i", current_edge_cost, this->agent->get_edge_progress(), this->agent->get_edge().x, this->agent->get_edge().y);
			//ROS_INFO("D_MCTS::search_from_root: got travel cost %0.2f", path_cost);
			this->distance = path_cost + (1 - this->agent->get_edge_progress()) * current_edge_cost;
			this->travel_time = this->distance / this->agent->get_travel_vel();
			//ROS_INFO("D_MCTS::search_from_root: distance %0.2f and travel time of %0.2f because travel vel %0.2f", this->distance, this->travel_time, this->agent->get_travel_vel());
			this->completion_time = this->world->get_c_time() + this->work_time + this->travel_time;
			//ROS_INFO("D_MCTS::search_from_root: completion_time %0.2f", this->completion_time);
			this->last_update_time = -INFINITY; // make sure I update the root every time
			this->get_expected_reward();
			//ROS_INFO("D_MCTS::search_from_root: completion_time %0.2f", this->completion_time);
		}
		else {
			ROS_ERROR("D_MCTS::search from root::failed to find travl_cost between %i -> %i", this->agent->get_edge().y, this->task_index);
			return;
		}
	}
	else {
		ROS_ERROR("D_MCTS::search from root::failed to find edge_cost for edge: %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
		return;
	}

	// I will be searched, count it!
	this->number_pulls++;

	if (this->kids.size() == 0) {
		if (!this->make_kids(task_status, task_set)) {
			//ROS_WARN("D_MCTS::search_from_root: could not make kids");
			return;
		}
	}

	D_MCTS* gc = NULL;
	//ROS_INFO("D_MCTS::search_from_root: has kids, searching");
	if (this->ucb_select_kid(gc)) {
		//ROS_INFO("D_MCTS::search_from_root: found ucb _kid");
		//ROS_ERROR("D_MCTS::search_from_root: found %i kids to search", int(this->kids.size()));
		//for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_ERROR("	D_MCTS::search_from_root: kids[%i]: %i", i, this->kids[i]->get_task_index());
		//}

		// search the kid's branch 
		int rollout_depth = -1;
		task_status[gc->get_task_index()] = false; // simulate completing the task
		gc->search(1, this->completion_time, task_status, task_set, rollout_depth);
		task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

		// do something with the reward
		this->update_branch_reward();
	}
}

void D_MCTS::search(const int &depth_in, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, int &rollout_depth) {
	
	//ROS_INFO("D_MCTS::Search: Searching: %i", this->task_index);
	if (task_status[this->task_index] == true) {
		ROS_ERROR("D_MCTS::search: bad task selected");
	}

	if (depth_in > this->max_search_depth || time_in > this->world->get_end_time() || rollout_depth > this->max_rollout_depth) {
		//ROS_ERROR("D_MCTS::search: returning");
		//ROS_ERROR("D_MCTS::search: depth in > this->max_search_depth: %i > %i", depth_in, this->max_search_depth);
		//ROS_ERROR("D_MCTS::search: time_in > this->world->get_end_time(): %0.2f > %0.2f", time_in, this->world->get_end_time());
		// if I am past the max search depth i have 0 search reward and should return without adding to passed branch reward
		return;
	}
	if(rollout_depth >= 0){
		// I am part of the rollout, increase the depth
		rollout_depth++;
	}
	
	// update with latest team member probable actions
	this->update_kid_rewards_with_new_probabilities();

	// I will be searched, count it!
	this->number_pulls++;

	// has my parents completion time moved? If yes, then I should update my arrival/completion time and then reward
	if ( abs(time_in - this->parent_time) > 0.2) {
		this->parent_time = time_in;
		this->distance = -1;
		this->get_expected_reward();
	}

	if (this->kids.size() == 0) {
		rollout_depth = 0;
		if (!this->make_kids(task_status, task_set)) {
			//ROS_WARN("D_MCTS::search_from_root: could not make kids");
			return;
		}
	}

	// if I have kids, then select kid with best search reward, and search them
	D_MCTS* gc = NULL;
	if (this->ucb_select_kid(gc)) {
		//ROS_ERROR("D_MCTS::search: found %i kids to search at depth %i", int(this->kids.size()), depth_in);
		//for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_ERROR("	D_MCTS::search: kids[%i]: %i", i, this->kids[i]->get_task_index());
		//}

		// search the kid's branch 
		task_status[gc->get_task_index()] = false; // simulate completing the task
		gc->search(depth_in + 1, this->completion_time, task_status, task_set, rollout_depth);
		task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

		// do something with the reward
		this->update_branch_reward();
	}
}

void D_MCTS::update_branch_reward() {
	this->max_kid_branch_reward = -INFINITY;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_branch_reward() > this->max_kid_branch_reward){
			this->max_kid_branch_reward = this->kids[i]->get_branch_reward();
			this->max_kid_index = i;
		}
	}
	this->branch_reward = this->expected_reward + this->max_kid_branch_reward;
}

void D_MCTS::get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards){
	path.push_back(this->task_index);
	times.push_back(this->completion_time);
	rewards.push_back(this->expected_reward);
	if(this->kids.size() > 0){
		int maxI = -1;
		double maxV = -INFINITY;
		for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_INFO("D_MCTS::get_best_path:: kid[%i] has task index %i and branch reward %0.1f", int(i), this->kids[i]->get_task_index(), this->kids[i]->get_branch_reward());
			if(this->kids[i]->get_branch_reward() > maxV){
				maxV = this->kids[i]->get_branch_reward();
				maxI = i;
			}
		}
		//ROS_WARN("D_MCTS::get_best_path:: maxI %i and maxV %0.1f for task %i", maxI, maxV, this->kids[maxI]->get_task_index());
		if(maxI >= 0){
			this->kids[maxI]->get_best_path(path, times, rewards);
		}
	}
}

bool D_MCTS::exploit_tree(int &max_kid_index, std::vector<std::string> &args, std::vector<double> &vals) {
	double maxV = -INFINITY;
	int maxI = -1;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_branch_reward() > maxV){
			maxV = this->kids[i]->get_branch_reward();
			maxI = i;
		}
	}

	if (maxI >= 0) {
		max_kid_index = maxI;

		args.push_back("distance");
		vals.push_back(this->kids[maxI]->get_distance());

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("completion_time");
		vals.push_back(this->kids[maxI]->get_completion_time());
		return true;
	}
	else {
		return false;
	}
}

void D_MCTS::prune_branches(const int &max_child) {
	for (size_t i = 0; i < this->kids.size(); i++) {
		if (i != max_child) {
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
	}
}

void D_MCTS::burn_branches() {
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->burn_branches();
		delete this->kids[i];
	}
}
