#include "Distributed_MCTS.h"
#include "World.h"
#include "Agent.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Pose.h"

#include <iostream>
#include "ros/ros.h"


Distributed_MCTS::Distributed_MCTS(World* world, Map_Node* task_in, Agent* agent_in, Distributed_MCTS* parent, const int &my_kid_index, const double &parent_time_in, const int update_index){
	this->agent = agent_in; // which agent am I?
	this->task = task_in; // which task am I?
	this->task_index = task_in->get_index(); // whats my index?
	this->world = world; // world
	

	this->last_update_index = update_index; // when was the last time my coordination info was updated
	
	if (parent) {
		this->parent = parent;
	}
	else {
		// Set as root if I don't have a parent
		this->my_probability = 1.0;
		this->parent_probability = 1.0;
		this->parent = NULL;
		if (task_in->is_active()) {
			// this is the root! I am already here!!!
			this->completion_time = parent_time_in + task_in->get_time_to_complete(this->agent, world);
		}
	}

	this->parent_time = parent_time_in;
	
	// MCTS stuff
	this->raw_reward = 0.0; // How much am I worth at completion time w/o coordination
	this->expected_reward = 0.0; // Including coordination, how much am I worth?
	this->down_branch_expected_reward = 0.0; // the expected reward for my best kids and their kids (recursive to max depth)
	this->number_pulls = 0; // how many times have I been pulled
	this->max_rollout_depth = 3; //edit these to ensure tree can grow on simple case
	this->max_search_depth = 20; 

	// sampling stuff
	this->alpha = 0.1; // Gradient ascent rate for coordination / learning rate on probable actions
	this->my_probability = 0.0; // How likely am I to be selected compared to siblings?
	this->parent_probability = 0.0; // How likely is my parent to be selected?
	this->min_sampling_threshold = 0.1; // How far down the tree will I search?

	// D-UCB Stuff
	this->cumulative_reward = 0.0; // what is the total cumulative reward across all historical pulls
	this->mean_reward = 0.0; // cumulative reward / n_pulls
	this->gamma = 0.99; // How long of a memory should I keep?
	this->epsilon = 1.41; // UCB weight term for parent n_pulls
	this->beta = 1.41; // explore vs exploit


	this->search_type = this->world->get_mcts_search_type();

	// initialize working variables
	this->last_update_index = update_index;
	this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	double dist = 0.0;
	std::vector<int> path;
	if (this->world->a_star(this->task_index, this->parent->get_task_index(), this->agent->get_pay_obstacle_cost(), false, path, dist)) {
		this->distance = dist;
		this->travel_time = this->distance / this->agent->get_travel_vel();
		this->completion_time = this->parent_time + this->travel_time + this->work_time;
		this->reward = this->task->get_reward_at_time(this->completion_time);
		this->update_my_probability();
	}
	else {
		this->expected_reward = -double(INFINITY);
		ROS_WARN("Distributed_MCTS::Initialize::Bad A* query");
	}
}


Distributed_MCTS::~Distributed_MCTS(){
	this->burn_branches();
}

bool Distributed_MCTS::make_kids( const std::vector<bool> &task_status, const std::vector<int> &task_set, const int &update_index ) {
	// task_set is a list of all available tasks
	// task status is a list of tasks current status
	// update index is when the coordination was last checked

	//ROS_WARN("Distributed_MCTS::make_kids: in with task_set.size(): %i", int(task_set.size()));
	// Make kids
	if (this->kids.size() > 0) {
		ROS_WARN("Distributed_MCTS::make_kids: already had kids????");
		for (size_t i = 0; i < this->kids.size(); i++) {
			ROS_WARN("Distributed_MCTS::make_kids: already had kids????");
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
		this->kids.clear();
	}

	bool kids_made = false;
	// potentially add a kid for each active task
	for (size_t j = 0; j < task_set.size(); j++) {
		//ROS_WARN("Distributed_MCTS::make_kids: checking kid[%i]", int(j));
		// which task am I looking at
		int ti = task_set[j]; // if task ti needs to be completed
		if (task_status[ti]) {
			//ROS_WARN("Distributed_MCTS::make_kids: task[%i] is active", int(j));
			Distributed_MCTS* kiddo = new Distributed_MCTS(this->world, this->world->get_nodes()[ti], this->agent, this, int(this->kids.size()), this->completion_time, update_index);
			this->kids.push_back(kiddo);
			kids_made = true;
		}
	}
	return kids_made;
}

bool Distributed_MCTS::ucb(Distributed_MCTS* &gc){
    //exactly the same as other ucb, but uses the historical mean value
    //of the cumulative reward to pick child. Works with D-UCB by adding a Gamma
    gc = NULL;
    if(this->kids.size() == 0){
    	return false;
    }

    this->n_pulls++;
    double minR = INFINITY;
    double maxR = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        this->kids[i].cum_reward = this->kids[i].cum_reward * this->gamma;
        this->kids[i].mean_reward = this->kids[i].cum_reward / std::max(0.01,this->kids[i].n_pulls);
        if(this->kids[i].mean_reward < minR){
            minR = this->kids[i].mean_reward;
        }
        if(this->kids[i].mean_reward > maxR){
            maxR = this->kids[i].mean_reward / std::max(0.01,this->kids[i].n_pulls);
        }
    }

    double maxM = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        double rr = (this->kids[i].mean_reward-minR) / std::max(0.01,(maxR-minR));
        this->kids[i].n_pulls = this->kids[i].n_pulls * this->kids[i].gamma;
        double ee = this->kids[i].beta*sqrt(this->kids[i].epsilon*log(this->n_pulls)/std::max(0.01,this->kids[i].n_pulls));
        if (rr + ee > maxM){
            maxM = rr + ee;
            gc = this->kids[i];
        }
    }
}

void Distributed_MCTS::search(bool &am_root, const int &depth_in, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, int &rollout_depth, const int &update_index) {
	
	//ROS_INFO("Distributed_MCTS::Search: Searching: %i", this->task_index);
	if (task_status[this->task_index] == true) {
		ROS_ERROR("Distributed_MCTS::search: bad task selected");
		this->expected_reward = 0.0;
		this->down_branch_expected_reward = 0.0;
		return;
	}

	if (depth_in > this->max_search_depth || time_in > this->world->get_end_time() || rollout_depth > this->max_rollout_depth) {
		this->down_branch_expected_reward = this->expected_reward;
		return;
	}
	
	if(rollout_depth >= 0){
		// I am part of the rollout, increase the depth
		rollout_depth++;
	}

	if (am_root && this->last_update_index != update_index){
		// I am the root and this is the first search of this iteration, check my travel time and update completion time
		this->travel_time = this->agent.get_travel_time(this->task_index);
		this->update_my_completion_time();
	}
	else if( fabs(time_in - this->parent_time) > 0.2) {
		// has my parents completion time moved? If yes, then I should update my arrival/completion time, probability, and then reward
		this->parent_time = time_in;
		this->last_update_index = update_index;
		this->update_my_completion_time();
	}

	if(this->last_update_index != update_index){
		// I recieved a coordination update, update my probability
		this->last_update_index = update_index;
		this->update_my_probability();
	}

	// If I don't have kids, make some and roll them out
	if (this->kids.size() == 0) {
		rollout_depth = std::max(0,rollout_depth);
		if (!this->make_kids(task_status, task_set, update_index)) {
			return;
		}
	}

	// if I have kids, then select kid with best search reward, and search them
	Distributed_MCTS* gc = NULL;
	if (this->ucb(gc)) {
		// search the kid's branch 
		task_status[gc->get_task_index()] = false; // simulate completing the task
		gc->search(false, depth_in + 1, this->completion_time, task_status, task_set, rollout_depth, update_index);
		task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

		// do something with the reward
		this->update_branch_reward(gc);
	}
}

void Distributed_MCTS::update_expected_branch_reward(Distributed_MCTS* &gc){
	// this checks all th kids to see who has the best expected reward and add it to my reward for my 
	// new best estiamte of my down branch reward. Then, it also updates the expected mean reward for 
	// the kid, gc, who was just searched
    double maxR = -INFINITY;
    double minR = INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        if(this->kids[i].down_branch_reward < minR){
            minR = this->kids[i].down_branch_reward;
        }
        if(this->kids[i].down_branch_reward > maxR){
            maxR = this->kids[i].down_branch_reward;
            this->max_kid = i;
        }
    }
    // update my down branch reward
    this->down_branch_expected_reward = this->my_reward + maxR;
    if(maxI > 0.0){
        // update searched kids cumulative reward
        this->kids[maxI].cum_reward = this->kids[maxI].cum_reward + (this->kids[maxI].down_branch_reward-minR) / std::max(0.001, maxR-minR);
    }
}

void Distributed_MCTS::sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in) {
	if (this->completion_time > this->world->get_end_time()) {
		return;
	}
	
	// add my task to coordinator
	coord_in->add_stop_to_my_path(this->task_index, this->completion_time, this->my_probability);
	if(this->kids.size() == 0){
		return;
	}
	// sample my children and assign probability
	this->find_kid_probabilities();
	// those kids who are good enough I should continue to sample
	for (size_t i = 0; i < this->kids.size(); i++) {
		//ROS_INFO("Distributed_MCTS::sample_tree_and_advertise_task_probabilities: kid[%i] probs: %0.2f and thresh is %0.2f", int(i), this->kids[i]->get_probability(), this->sampling_probability_threshold);
		if (this->kids[i]->get_probability() > this->sampling_probability_threshold) {
			//std::cerr << "Distributed_MCTS::sample_tree_and_advertise_task_probabilities: this->kids[]->get_task_index(): " << this->kids[i]->get_task_index() << std::endl;
			this->kids[i]->sample_tree_and_advertise_task_probabilities(coord_in);
		}
	}
}

void Distributed_MCTS::update_probable_actions() {
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
		//ROS_INFO("Distributed_MCTS::sample_tree_and_advertise_task_probabilities: kid[%i] probs: %0.2f and thresh is %0.2f", int(i), this->kids[i]->get_probability(), this->sampling_probability_threshold);
		if (this->kids[i]->get_probability() > this->sampling_probability_threshold) {
			//std::cerr << "Distributed_MCTS::sample_tree_and_advertise_task_probabilities: this->kids[]->get_task_index(): " << this->kids[i]->get_task_index() << std::endl;
			this->kids[i]->update_probable_actions();
		}
	}
}

void Distributed_MCTS::find_kid_probabilities() {
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
			double p = this->my_probability * (this->kids[i]->get_branch_reward() / sumR);
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
		//ROS_WARN("Distributed_MCTS::find_kid_probabilities: maxK: %i (task %i) with maxR: %0.1f", maxK, this->kids[maxK]->get_task_index(), maxR);
		// Adjust probabilities
		for (size_t i=0; i<this->kids.size(); i++){
			//ROS_WARN("Distributed_MCTS::find_kid_probabilities: p_init[%i]: %0.2f", int(i), this->kids[i]->get_probability());
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
			//ROS_WARN("Distributed_MCTS::find_kid_probabilities: p_final[%i]: %0.2f", int(i), this->kids[i]->get_probability());
			pSum += this->kids[i]->get_probability();
		}
		// Normalize Probabilities
		for(size_t i=0; i<this->kids.size(); i++){
			double p = this->kids[i]->get_probability() / pSum;
			this->kids[i]->set_probability( this->my_probability * p );
		}
	}
}

double Distributed_MCTS::get_branch_reward() {
	if (this->kids.size() == 0) {
		this->branch_reward = this->get_expected_reward();
	}
	else {
		this->branch_reward = this->get_expected_reward() + this->max_kid_branch_reward;
	}
	return this->branch_reward;
}

void Distributed_MCTS::reset_mcts_team_prob_actions(){
	this->reset_task_availability_probability();
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->reset_task_availability_probability();
	}	
}

void Distributed_MCTS::update_my_completion_time(){
	// new arrival time, already know distance
	this->completion_time = this->parent_time + this->work_time;
	this->reward = this->task->get_reward_at_time(this->completion_time);
	this->update_my_probability();
}

void Distributed_MCTS::update_my_probability(){
	// Update probability and everything down stream of it
	double p_taken = 0.0;
	if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
		this->probability_task_available = (1 - p_taken);
		this->expected_reward *= this->probability_task_available;
	}
}

void Distributed_MCTS::update_branch_reward() {
	this->max_kid_branch_reward = -INFINITY;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_branch_reward() > this->max_kid_branch_reward){
			this->max_kid_branch_reward = this->kids[i]->get_branch_reward();
			this->max_kid_index = i;
		}
	}
	this->branch_reward = this->expected_reward + this->max_kid_branch_reward;
}

void Distributed_MCTS::get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards){
	path.push_back(this->task_index);
	times.push_back(this->completion_time);
	rewards.push_back(this->expected_reward);
	if(this->kids.size() > 0){
		int maxI = -1;
		double maxV = -INFINITY;
		for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_INFO("Distributed_MCTS::get_best_path:: kid[%i] has task index %i and branch reward %0.1f", int(i), this->kids[i]->get_task_index(), this->kids[i]->get_branch_reward());
			if(this->kids[i]->get_branch_reward() > maxV){
				maxV = this->kids[i]->get_branch_reward();
				maxI = i;
			}
		}
		//ROS_WARN("Distributed_MCTS::get_best_path:: maxI %i and maxV %0.1f for task %i", maxI, maxV, this->kids[maxI]->get_task_index());
		if(maxI >= 0){
			this->kids[maxI]->get_best_path(path, times, rewards);
		}
	}
}

bool Distributed_MCTS::exploit_tree(int &max_kid_index, std::vector<std::string> &args, std::vector<double> &vals) {
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

void Distributed_MCTS::prune_branches(const int &max_child) {
	for (size_t i = 0; i < this->kids.size(); i++) {
		if (i != max_child) {
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
	}
}

void Distributed_MCTS::burn_branches() {
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->burn_branches();
		delete this->kids[i];
	}
}
