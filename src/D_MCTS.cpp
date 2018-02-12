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
		this->probability = 0.0;
		this->kid_index = my_kid_index;
		this->max_kid_distance_threshold = parent->get_max_kid_distance_threshold();
		this->parent = parent;
		this->completion_time = -1.0;
	}
	else {
		// Set as root if I don't have a parent
		this->probability = 1.0;
		this->kid_index = -1;
		this->max_kid_distance_threshold = 200000.0;// sqrt(pow(world->get_height(), 2) + pow(world->get_width(), 2)); // how far can I travel to a child
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
	this->expected_value = 0.0; // expected reward of this child

	// task stuff
	this->probability_task_available = 1.0;
	this->work_time = -1.0;
	
	// MCTS stuff
	this->rollout_reward = 0.0;
	this->branch_value = -1.0; // my expected value + all my best kids expected value
	this->explore_value = -1.0; // haven't searched in a while value
	this->search_value = -1.0; // = explore_value + exploit_value
	this->number_pulls = 0; // how many times have I been pulled
	this->max_rollout_depth = 3; //edit these to ensure tree can grow on simple case
	this->max_search_depth = 50; 

	// sampling stuff
	this->max_kid_index = -1; // index of gc
	this->min_kid_index = -1;
	this->max_kid_branch_value = -1.0; // their value
	this->min_kid_branch_value = -1.0;
	this->sum_kid_branch_value = 0.0;

	//std::cerr << "c" << std::endl;
	// few useful constants
	if (this->world->get_task_selection_method() == "mcts_task_by_completion_reward") {
		this->reward_weighting = 1.0; // how important is the reward in the value function
		this->distance_weighting = 0.01; // how important is the travel cost in the value function
	}
	else if (this->world->get_task_selection_method() == "mcts_task_by_completion_value") {
		this->reward_weighting = 1.0; // how important is the reward in the value function
		this->distance_weighting = 1.0; // how important is the travel cost in the value function
	}
	else if (this->world->get_task_selection_method() == "mcts_task_by_completion_reward_impact") {
		this->reward_weighting = 1.0; // how important is the reward in the value function
		this->distance_weighting = 0.01; // how important is the travel cost in the value function
	}

	//std::cerr << "d" << std::endl;

	this->search_type = this->world->get_mcts_search_type();
	this->alpha = 0.05; // gradient descent rate, how fast should my team 
	this->beta = 0.0705; // this is what I found in Matlab tests//20.0; //1.41; // ucb = 1.41, d-ucb = 1.41, sw-ucb = 0.705
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

void D_MCTS::find_min_branch_value_kid() {
	// find the kid with minimum expected value
	this->min_kid_branch_value = double(INFINITY);
	this->min_kid_index = -1;
	
	for (size_t i = 0; i < this->kids.size(); i++) {
		if (this->kids[i]->get_branch_value() < this->min_kid_branch_value) {
			this->min_kid_branch_value = this->kids[i]->get_branch_value();
			this->min_kid_index = int(i);
		}
	}
}

void D_MCTS::find_max_branch_value_kid() {
	// find the kid with the maximum expected value
	this->max_kid_branch_value = -double(INFINITY);
	this->max_kid_index = -1;
	
	for (size_t i = 0; i < this->kids.size(); i++) {
		if (this->kids[i]->get_branch_value() > this->max_kid_branch_value) {
			this->max_kid_branch_value = this->kids[i]->get_branch_value();
			this->max_kid_index = int(i);
		}
	}
}

void D_MCTS::find_sum_kid_branch_value() {
	// find the kid with the maximum expected value
	this->sum_kid_branch_value = 0.0;

	for (size_t i = 0; i < this->kids.size(); i++) {
		this->sum_kid_branch_value += this->kids[i]->get_branch_value();
	}
}

D_MCTS::~D_MCTS(){}


void D_MCTS::set_as_root() {
	this->probability = 1.0;
}

void D_MCTS::sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in) {
	
	if (this->completion_time > this->world->get_end_time()) {
		return;
	}


	if (this->probability < this->sampling_probability_threshold) {
		// this should never happen, due to below, but why not double check?
		return;
	}
	else {
		// add my task to coordinator
		coord_in->add_stop_to_my_path(this->task_index, this->completion_time, this->probability);
		// sample my children and assign probability
		this->find_kid_probabilities();

		// those kids who are good enough I should continue to sample
		for (size_t i = 0; i < this->kids.size(); i++) {
			if (this->kids[i]->get_probability() > this->sampling_probability_threshold) {
				this->kids[i]->sample_tree_and_advertise_task_probabilities(coord_in);
			}
		}
	}
}

void D_MCTS::find_kid_probabilities() {

	// for all kids, assign their probability
	if(this->world->get_impact_style() != "gradient_descent"){
		// none of these should be true, but just check
		if (this->sum_kid_branch_value < 0.0) {
			this->find_sum_kid_branch_value();
		}
		if (this->sum_kid_branch_value == 0) {
			return;
		}
		if (this->max_kid_branch_value < 0.0) {
			this->find_max_branch_value_kid();
		}
		if (this->min_kid_branch_value < 0.0) {
			this->find_min_branch_value_kid();
		}
		for (size_t i = 0; i < this->kids.size(); i++) {
			//TODO this->kids[i]->set_probability(this->sum_kid_branch_value, this->probability);
			this->kids[i]->set_probability(this->sum_kid_branch_value, this->min_kid_branch_value, this->max_kid_branch_value, this->probability);
		}
	}
	else{
		// Use Gradient descent to get kid probabilities

		// Get max kid
		int maxK = -1;
		double maxV = -INFINITY;
		for (size_t i = 0; i < this->kids.size(); i++) {
			if(this->kids[i]->get_branch_value() > maxV){
				maxV = this->kids[i]->get_branch_value();
				maxK = i;
			}
		}
		double pSum = 0.0;
		// Adjust probabilities
		for (size_t i=0; i<this->kids.size(); i++){
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
			pSum += this->kids[i]->get_probability();
		}
		// Normalize Probabilities
		for(size_t i=0; i<this->kids.size(); i++){
			this->kids[i]->set_probability( this->kids[i]->get_probability() / pSum );
		}
	}
}

void D_MCTS::set_probability(const double &sum, const double &min, const double &max, const double &parent_probability) {
	// this sets my probability of being selected by my parent
	
	//double wv = (this->branch_value - min) / (max - min);
	double wv = this->branch_value / sum;
	double ov = 0.0;
	if (this->branch_value == max) {
		ov = 1.0;
	}

	if (this->world->get_impact_style() == "optimal") {
		this->probability = parent_probability * ov + (1-parent_probability)*wv;
	}
	else if (this->world->get_impact_style() == "fixed") {
		this->probability = 0.5 * ov + (1 - 0.5) *wv;
		this->probability *= parent_probability;
	}
	else {
		double p = std::min(0.95, parent_probability);
		this->probability = p * ov + (1 - p) *wv;
	}
}

void D_MCTS::set_probability(const double &sum_value, const double &parent_probability) {
	// this sets my probability of being selected by my parent
	this->probability = parent_probability * this->branch_value / sum_value;
}

void D_MCTS::rollout(const int &c_index, const int &rollout_depth, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, double &passed_branch_value) {
	if (rollout_depth > this->max_rollout_depth || time_in > this->world->get_end_time()) {
		return;
	}

	int max_index = -1;
	double max_comp_reward = 0.0; // I only want positive rewards!
	double max_comp_time = 0.0;

	for (size_t j = 0; j < task_set.size(); j++) { // this intentionally does not use kids to rollout, rolled out nodes don't have kids and I probably don't want to make them yet.
		int i = task_set[j];
		if (task_status[i]) {
			double e_dist = double(INFINITY);
			// get euclidean dist first
			if (world->get_travel_cost(c_index, int(i), this->agent->get_pay_obstacle_cost(), e_dist) && e_dist < this->max_kid_distance_threshold) {
				// this is for all tasks, not kids so need to do these
				double e_time = e_dist / this->agent->get_travel_vel();
				double w_time = world->get_nodes()[i]->get_time_to_complete(this->agent, world);
				double comp_time = time_in + e_time + w_time;
				double e_reward = world->get_nodes()[i]->get_reward_at_time(world->get_c_time() + comp_time);
				// is my euclidean travel time reward better?
				if (e_reward > max_comp_reward) {
					// I am euclidean reward better, check if it is taken by someone else?
					double prob_taken = 0.0;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(int(i), comp_time, prob_taken, world)) {
						//if(prob_taken > 0.0){
						//	ROS_ERROR("D_MCTS::rollout: p_taken > 0");
						//}
						//else{
						//	ROS_WARN("D_MCTS::rollout: p_taken == 0");	
						//}
						// if not taken, then accept as possible goal
						if ((1 - prob_taken)*e_reward > max_comp_reward) {
							max_comp_reward = (1 - prob_taken)*e_reward;
							max_index = int(i);
							max_comp_time = comp_time;
						}
					}
				}
			}
		}
	}
	if (max_index > -1) {
		// found the next kid to rollout, add kids value found in search and search kid.
		passed_branch_value += max_comp_reward; // add this iteration's reward

		// search below
		task_status[max_index] = false; // set task I am about to rollout as complete
		rollout(max_index, rollout_depth + 1, max_comp_time, task_status, task_set, passed_branch_value); // rollout selected task
		task_status[max_index] = true; // reset task I just rolledout

		return;
	}
	else {
		return;
	}
}

bool D_MCTS::ucb_select_kid(D_MCTS* &gc) {
	
	gc = NULL;
	double maxV = -INFINITY;
	double minV = INFINITY;
	

	for (size_t i = 0; i < this->kids.size(); i++) { // check all of my kids
		if ( this->kids[i]->get_branch_value() > maxV) { // is their a* value better than maxval?				
			maxV = this->kids[i]->get_branch_value();
		}
		if (this->kids[i]->get_branch_value() < minV){
			minV = this->kids[i]->get_branch_value();
		}
	}
	double dV = maxV - minV;
	double maxM = -1;
	if(maxV > 0){
		for(size_t i=0; i<this->kids.size(); i++){
			double rr = (this->kids[i]->get_branch_value() - minV) / dV;
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

double D_MCTS::get_branch_value() {
	if (this->kids.size() == 0) {
		this->branch_value = this->get_expected_value();
	}
	else {
		this->branch_value = this->get_expected_value() + this->max_kid_branch_value;
	}
	return this->branch_value;
}

double D_MCTS::get_search_value(const double &min, const double &max, const int &planning_iter) {
	// remember, this is for ME, not from my parent, all internal

	if (this->search_type == "UCT") {
		// this is the expected value normalized in range of kid values
		this->get_branch_value();
		if (this->branch_value > 0 && max - min > 0) {
			this->exploit_value = (this->branch_value - min) / (max - min);
		}
		else {
			this->exploit_value = 0.0;
		}

		if (this->explore_value == -1.0) {
			this->explore_value = this->beta*sqrt(this->epsilon*log(this->parent->get_n_pulls()) / std::max(0.0001, this->number_pulls));
		}

		this->search_value = this->exploit_value + this->explore_value;
		return this->search_value;

	}
	if (this->search_type == "SW-UCT") {
		// remove elements not in window!
		while (this->time_log.size() > 0 && this->time_log[0] + this->window_width < planning_iter) {
			// remove value for elements outside of window
			this->sum_exp_value -= this->exp_value_log[0] * pow(this->gamma, this->time_log.back() - this->time_log[0]);
			this->sum_n_pulls -= 1.0 * pow(this->gamma, this->time_log.back() - this->time_log[0]);

			// remove elements
			this->time_log.erase(this->time_log.begin());
			this->exp_value_log.erase(this->exp_value_log.begin());
		}
		
		// get exploit value
		// include the current guesss at value, in case I know that it is completed
		this->get_branch_value();
		this->exploit_value = 0.0;
		if (this->branch_value > 0 && max - min > 0) {
			this->exploit_value = (this->branch_value - min) / (max - min);
		}
		this->exploit_value += this->sum_exp_value;
		this->exploit_value = this->exploit_value / (1 + this->sum_n_pulls);
		
		// get explore value
		if (time_log.size() > 0) {
			double temp_n_pulls = this->sum_n_pulls * pow(this->gamma, planning_iter - this->time_log.back());
			this->explore_value = this->beta*sqrt(this->epsilon*log(std::min(double(this->window_width), this->parent->get_n_pulls())) / std::max(0.0001, temp_n_pulls));
		}
		else {
			this->explore_value = this->beta*sqrt(this->epsilon*log(std::min(double(this->window_width), this->parent->get_n_pulls())) / 0.0001);
		}
		this->search_value = this->exploit_value + this->explore_value;
		return this->search_value;

	}
}

void D_MCTS::add_sw_uct_update(const double &min, const double &max, const int &planning_iter) {
	// update exploit value
	double t_exp_value = 0.0;
	if (this->branch_value > 0 && max - min > 0) {
		t_exp_value = (this->branch_value - min) / (max - min);
	}
	if (this->time_log.size() > 0) {
		// decrement old exploitvalues
		this->sum_exp_value = pow(this->gamma, planning_iter - this->time_log.back()) * this->sum_exp_value;
	}
	// add new exploit value and log it
	this->sum_exp_value += t_exp_value;
	this->exp_value_log.push_back(t_exp_value);
	
	// update pull value
	if (this->time_log.size() > 0) {
		this->sum_n_pulls = pow(this->gamma, planning_iter - this->time_log.back()) * this->sum_n_pulls;
	}
	this->sum_n_pulls += 1.0;
	this->time_log.push_back(planning_iter);
}

double D_MCTS::get_expected_value() {
	// already know task is incomplete, this is only the value for completing me!
	// DOES NOT INCLUDE KIDS! NOT BRANCH VALUE!

	if (this->work_time < 0.0) {
		this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	}

	// being set for the first time?
	if (this->distance < 0.0) {
		this->last_update_time = this->world->get_c_time();
		// need to set everything, then return value
		double dist = 0.0;
		std::vector<int> path;
		if (this->world->a_star(this->task_index, this->parent->get_task_index(), this->agent->get_pay_obstacle_cost(), path, dist)) {
			this->distance = dist;
			this->travel_time = this->distance / this->agent->get_travel_vel();
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			//ROS_INFO("initialized node: %i with completion_time: %0.1f", this->task_index, this->completion_time);
			if (this->world->get_mcts_reward_type() == "impact") {
				this->reward = this->agent->get_coordinator()->get_reward_impact(this->task_index, this->agent->get_index(), this->completion_time, this->world);
				//ROS_INFO("has impact reward %0.2f", this->reward);
			}
			else {
				this->reward = this->task->get_reward_at_time(this->completion_time);
				//ROS_INFO("has reward %0.2f", this->reward);
				double p_taken = 0.0;
				if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
					//if(p_taken > 0.0){
					//	ROS_ERROR("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
					//}
					//else{
					//	ROS_WARN("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
					//}
					this->probability_task_available = (1 - p_taken);
					this->reward *= this->probability_task_available;
					//ROS_INFO("has reward %0.2f", this->reward);
				}
			}
			this->expected_value = this->reward_weighting*this->reward - this->distance_weighting*this->distance;
			//ROS_INFO("has expected_value %0.2f", this->expected_value);
			//ros::Duration d = ros::Duration(1.0);
			//d.sleep();
		}
		else {
			this->expected_value = -double(INFINITY);
			std::cout << "mcts::bad A* query" << std::endl;
		}

		return this->expected_value;
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
		if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->travel_time, p_taken, this->world)) {
			//if(p_taken > 0.0){
			//	ROS_ERROR("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
			//}
			//else{
			//	ROS_WARN("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
			//}
			this->probability_task_available = (1 - p_taken);
			this->reward *= this->probability_task_available;
		}
		this->expected_value = this->reward_weighting*this->reward - this->distance_weighting*this->distance;
		return this->expected_value;
	}

	// next chance is that the probabilities changed because of a report
	if (this->world->get_c_time() - this->last_update_time > 1.0) {
		this->last_update_time = this->world->get_c_time();
		double p_taken = 0.0;
		this->reward = this->task->get_reward_at_time(this->completion_time);
		if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
			//if(p_taken > 0.0){
			//	ROS_ERROR("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f > 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);
			//}
			//else{
			//	ROS_WARN("D_MCTS::get_expected_value:[%i] p_taken[%i]=%0.1f == 0 at time: %0.2f", this->agent->get_index(), this->task_index, p_taken, this->completion_time);	
			//}
			this->probability_task_available = (1 - p_taken);
			this->reward *= this->probability_task_available;
		}
		this->expected_value = this->reward_weighting*this->reward - this->distance_weighting*this->distance;

		return this->expected_value;
	}

	// don't need to set value, just return
	return this->expected_value;
}

void D_MCTS::update_kid_values_with_new_probabilities() {
	// need to update each kids expected value and then get their updated branch value
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->reset_task_availability_probability();
		this->kids[i]->get_expected_value();
		this->kids[i]->get_branch_value();
	}
}

void D_MCTS::reset_mcts_team_prob_actions(){
	this->reset_task_availability_probability();
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->reset_task_availability_probability();
	}	
}

void D_MCTS::search_from_root(std::vector<bool> &task_status, std::vector<int> &task_set, const int &last_planning_iter_end, const int &planning_iter) {
	//ROS_ERROR("D_MCTS::search_from_root: in");
	//make sure work time is set
	if (this->work_time < 0.0) {
		this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	}

	// should I reset probabilities?
	if (this->last_planning_iter_end < last_planning_iter_end) {
		this->last_planning_iter_end = last_planning_iter_end;
		this->update_kid_values_with_new_probabilities();
	}

	// make sure completion and travel time are correct
	double path_cost = 0.0;
	double current_edge_cost = 0.0;

	if (this->world->get_edge_cost(this->agent->get_edge().x, this->agent->get_edge().y, this->agent->get_pay_obstacle_cost(), current_edge_cost)) {
		if (this->world->get_travel_cost(this->agent->get_edge().y, this->task_index, this->agent->get_pay_obstacle_cost(), path_cost)) {
			this->distance = path_cost + (1 - this->agent->get_edge_progress()) * current_edge_cost;
			this->travel_time = this->distance / this->agent->get_travel_vel();
			this->completion_time = this->world->get_c_time() + this->work_time + this->travel_time;
			this->last_update_time = -INFINITY; // make sure I update the root every time
			//this->get_expected_value();
		}
		else {
			ROS_ERROR("D_MCTS::search from root::failed to find edge_cost");
			return;
		}
	}
	else {
		ROS_ERROR("D_MCTS::search from root::failed to find travel_cost");
		return;
	}

	// I will be searched, count it!
	this->number_pulls++;

	if (this->kids.size() == 0) {
		if (!this->make_kids(task_status, task_set)) {
			ROS_WARN("D_MCTS::search_from_root: could not make kids");
		}
	}

	if(this->kids.size() > 0){
		D_MCTS* gc = NULL;
		if (this->ucb_select_kid(gc)) {
			//ROS_ERROR("D_MCTS::search_from_root: found %i kids to search", int(this->kids.size()));
			//for(size_t i=0; i<this->kids.size(); i++){
			//	ROS_ERROR("	D_MCTS::search_from_root: kids[%i]: %i", i, this->kids[i]->get_task_index());
			//}

			// search the kid's branch 
			task_status[gc->get_task_index()] = false; // simulate completing the task
			gc->search(1, this->completion_time, task_status, task_set, last_planning_iter_end, planning_iter);
			task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

			// do something with the reward
			this->update_branch_value();
		}
	}
}

void D_MCTS::search(const int &depth_in, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, const int &last_planning_iter_end, const int &planning_iter) {
	
	//ROS_INFO("D_MCTS::Search: Searching: %i", this->task_index);
	if (task_status[this->task_index] == true) {
		ROS_ERROR("D_MCTS::search: bad task selected");
	}

	if (depth_in > this->max_search_depth || time_in > this->world->get_end_time()) {
		//ROS_ERROR("D_MCTS::search: returning");
		//ROS_ERROR("D_MCTS::search: depth in > this->max_search_depth: %i > %i", depth_in, this->max_search_depth);
		//ROS_ERROR("D_MCTS::search: time_in > this->world->get_end_time(): %0.2f > %0.2f", time_in, this->world->get_end_time());
		// if I am past the max search depth i have 0 search reward and should return without adding to passed branch value
		return;
	}
	
	// should I reset probabilities?
	if (this->last_planning_iter_end < last_planning_iter_end) {
		this->last_planning_iter_end = last_planning_iter_end;
		this->update_kid_values_with_new_probabilities();
	}
	// I will be searched, count it!
	this->number_pulls++;
	this->explore_value = -1.0; // reset explore value so it is recomputed next iter, this is implemented in get_explore_value()

	if ( abs(time_in - this->parent_time) > 0.2) {
		this->parent_time = time_in;
		this->distance = -1;
		this->get_expected_value();
	}

	if (this->kids.size() > 0) {

		// get kis who no longer have active tasks
		this->erase_null_kids();

		// if I have kids, then select kid with best search value, and search them
		D_MCTS* gc = NULL;
		if (this->ucb_select_kid(gc)) {
			//ROS_ERROR("D_MCTS::search: found %i kids to search at depth %i", int(this->kids.size()), depth_in);
			//for(size_t i=0; i<this->kids.size(); i++){
			//	ROS_ERROR("	D_MCTS::search: kids[%i]: %i", i, this->kids[i]->get_task_index());
			//}

			// search the kid's branch 
			task_status[gc->get_task_index()] = false; // simulate completing the task
			gc->search(depth_in + 1, this->completion_time, task_status, task_set, last_planning_iter_end, planning_iter);
			task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

			// do something with the reward
			this->update_branch_value();
		}
	}
}

void D_MCTS::update_branch_value() {
	this->max_kid_branch_value = -INFINITY;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_branch_value() > this->max_kid_branch_value){
			this->max_kid_branch_value = this->kids[i]->get_branch_value();
		}
	}
	this->branch_value = this->expected_value + this->max_kid_branch_value;
}

void D_MCTS::get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards){
	path.push_back(this->task_index);
	times.push_back(this->completion_time);
	rewards.push_back(this->expected_value);
	if(this->kids.size() > 0 && this->max_kid_index >= 0){
		this->kids[this->max_kid_index]->get_best_path(path, times, rewards);

	}
}

bool D_MCTS::exploit_tree(int &goal_index, std::vector<std::string> &args, std::vector<double> &vals) {

	if (this->max_kid_index >= 0) {
		goal_index = this->kids[this->max_kid_index]->get_task_index();

		args.push_back("distance");
		vals.push_back(this->kids[this->max_kid_index]->get_distance());

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("completion_time");
		vals.push_back(this->kids[this->max_kid_index]->get_completion_time());
		return true;
	}
	else {
		return false;
	}
}

void D_MCTS::burn_branches() {
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->burn_branches();
		delete this->kids[i];
	}
}

D_MCTS* D_MCTS::get_golden_child() {
	return this->kids[this->max_kid_index];
}
