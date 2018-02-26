#include "Distributed_MCTS.h"
#include "World.h"
#include "Agent.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Pose.h"

#include <iostream>
#include "ros/ros.h"


Distributed_MCTS::Distributed_MCTS(World* world, Map_Node* task_in, Agent* agent_in, Distributed_MCTS* parent, const int &my_kid_index, const int update_index){
	ROS_WARN("Distributed_MCTS::Distributed_MCTS: initializing Distributed_MCTS class");
	this->agent = agent_in; // which agent am I?
	this->task = task_in; // which task am I?
	this->task_index = task_in->get_index(); // whats my index?
	this->world = world; // world
	

	this->last_update_index = update_index; // when was the last time my coordination info was updated
	
	if (parent) {
		this->parent = parent;
		this->parent_time = parent->get_completion_time();
		this->parent_index = parent->get_task_index();
	}
	else {
		// Set as root if I don't have a parent
		this->raw_probability = 1.0;
		this->branch_probability = 1.0;
		this->parent = NULL;
		if (task_in->is_active()) {
			// this is the root! I am already here!!!
			this->completion_time = this->world->get_c_time() + task_in->get_time_to_complete(this->agent, world);
		}
	}
	
	// MCTS stuff
	this->raw_reward = 0.0; // How much am I worth at completion time w/o coordination
	this->expected_reward = 0.0; // Including coordination, how much am I worth?
	this->down_branch_expected_reward = 0.0; // the expected reward for my best kids and their kids (recursive to max depth)
	this->number_pulls = 0; // how many times have I been pulled
	this->max_rollout_depth = 3; //edit these to ensure tree can grow on simple case
	this->max_search_depth = 10; 

	// sampling stuff
	this->alpha = 0.1; // Gradient ascent rate for coordination / learning rate on probable actions
	this->raw_probability = 0.0; // How likely am I to be selected compared to siblings?
	this->branch_probability = 0.0; // How likely am I to be selected accounting for parent
	this->min_sampling_probability_threshold = 0.1; // How far down the tree will I search?

	// D-UCB Stuff
	this->cumulative_reward = 0.0; // what is the total cumulative reward across all historical pulls
	this->mean_reward = 0.0; // cumulative reward / number_pulls
	this->gamma = 0.99; // How long of a memory should I keep?
	this->epsilon = 1.41; // UCB weight term for parent number_pulls
	this->beta = 1.41; // explore vs exploit

	this->kids.clear();

	//ROS_WARN("Distributed_MCTS::Distributed_MCTS: setting initial values");
	this->search_type = this->world->get_mcts_search_type();
	//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got search type");
	// initialize working variables
	this->last_update_index = update_index;
	this->work_time = this->task->get_time_to_complete(this->agent, this->world);
	if(parent){
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got work time");
		double dist = 0.0;
		std::vector<int> path;
		if (this->world->a_star(this->task_index, this->parent->get_task_index(), this->agent->get_pay_obstacle_cost(), false, path, dist)) {
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: A* complete");
			this->distance = dist;
			this->travel_time = this->distance / this->agent->get_travel_vel();
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: travel time complete");
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: completion time complete");
			this->raw_reward = this->task->get_reward_at_time(this->completion_time);
			this->update_probability_task_is_available();
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got probability");
			this->expected_reward = this->raw_reward * (1.0 - this->probability_task_available);
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got expected_reward");
		}

		else {
			this->expected_reward = -double(INFINITY);
			//ROS_WARN("Distributed_MCTS::Initialize::Bad A* query");
		}
	}
	else{
		if(this->agent->get_travel_time(this->task_index, this->travel_time)){
			// I am the root, get time from me to task
			this->completion_time = this->world->get_c_time() + this->travel_time + this->work_time;
			//ROS_ERROR("Distributed_MCTS::search: got completion time: %0.2f", this->completion_time);
		}
	}
}


Distributed_MCTS::~Distributed_MCTS(){
//	this->burn_branches();
}

bool Distributed_MCTS::make_kids( const std::vector<bool> &task_status, const std::vector<int> &task_set, const int &update_index ) {
	// task_set is a list of all available tasks
	// task status is a list of tasks current status
	// update index is when the coordination was last checked

	//ROS_WARN("Distributed_MCTS::make_kids: in with task_set.size(): %i", int(task_set.size()));
	// Make kids
	bool kids_made = false;
	// potentially add a kid for each active task
	for (size_t j = 0; j < task_set.size(); j++) {
		//ROS_WARN("Distributed_MCTS::make_kids: checking kid[%i]", int(j));
		// which task am I looking at
		int ti = task_set[j]; // if task ti needs to be completed
		if (task_status[ti]) {
			//ROS_WARN("Distributed_MCTS::make_kids: task[%i] is active", int(j));
			Distributed_MCTS* kiddo = new Distributed_MCTS(this->world, this->world->get_nodes()[ti], this->agent, this, int(this->kids.size()), update_index);
			this->kids.push_back(kiddo);
			kids_made = true;
		}
	}

	if(kids_made){
		this->perform_initial_sampling();
	}

	return kids_made;
}

void Distributed_MCTS::perform_initial_sampling(){
	// perform initial sampling of kids
    double sumR = 0;
    for(size_t i=0; i<this->kids.size(); i++){
        sumR += this->kids[i]->get_down_branch_expected_reward();
    }
    // minR / maxR does NOT work, puts all 0 to 1 individually, allowing multiple to be 1.0
    for(size_t i=0; i<this->kids.size(); i++){
        this->kids[i]->set_raw_probability(this->kids[i]->get_down_branch_expected_reward()/sumR);
        this->kids[i]->set_branch_probability(this->branch_probability * this->kids[i]->raw_probability);
    }
}

bool Distributed_MCTS::ucb(Distributed_MCTS* &gc){
    //exactly the same as other ucb, but uses the historical mean value
    //of the cumulative reward to pick child. Works with D-UCB by adding a Gamma
    gc = NULL;
    if(this->kids.size() == 0){
    	return false;
    }

    this->number_pulls++;
    double minR = INFINITY;
    double maxR = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        this->kids[i]->cumulative_reward = this->kids[i]->cumulative_reward * this->gamma;
        this->kids[i]->mean_reward = this->kids[i]->cumulative_reward / std::max(0.01,this->kids[i]->number_pulls);
        if(this->kids[i]->mean_reward < minR){
            minR = this->kids[i]->mean_reward;
        }
        if(this->kids[i]->mean_reward > maxR){
            maxR = this->kids[i]->mean_reward / std::max(0.01,this->kids[i]->number_pulls);
        }
    }

    double maxM = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        double rr = (this->kids[i]->mean_reward-minR) / std::max(0.01,(maxR-minR));
        this->kids[i]->number_pulls = this->kids[i]->number_pulls * this->kids[i]->gamma;
        double ee = this->kids[i]->beta*sqrt(this->kids[i]->epsilon*log(this->number_pulls)/std::max(0.01,this->kids[i]->number_pulls));
        if (rr + ee > maxM){
            maxM = rr + ee;
            gc = this->kids[i];
        }
    }
    return true;
}

void Distributed_MCTS::search(const bool &am_root, int depth_in, Distributed_MCTS* parent, std::vector<bool> &task_status, std::vector<int> &task_set, int &rollout_depth, const int &update_index) {
	ROS_ERROR("into search at depth: %i", depth_in);
	//ROS_INFO("Distributed_MCTS::Search: Searching: %i", this->task_index);
	if (task_status[this->task_index] == true) {
		ROS_ERROR("Distributed_MCTS::search: bad task selected");
		this->expected_reward = 0.0;
		this->down_branch_expected_reward = 0.0;
		return;
	}

	if (!am_root && (depth_in > this->max_search_depth || parent->get_completion_time() > this->world->get_end_time() || rollout_depth > this->max_rollout_depth)) {
		this->down_branch_expected_reward = -INFINITY;
		this->raw_reward = -INFINITY;
		return;
	}
	else{
		depth_in++;
	}
	
	if(rollout_depth >= 0){
		// I am part of the rollout, increase the depth
		rollout_depth++;
	}

	if (am_root && this->last_update_index != update_index){
		// I am the root and this is the first search of this iteration, check my travel time and update completion time
		//ROS_ERROR("Distributed_MCTS::search: getting completion time, currently it is: %0.2f", this->completion_time);
		//ROS_ERROR("Distributed_MCTS::search: 	for goal: %i", this->task_index);
		if(this->agent->get_travel_time(this->task_index, this->travel_time)){
			this->parent_time = this->world->get_c_time();
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			//ROS_ERROR("Distributed_MCTS::search: got completion time: %0.2f", this->completion_time);
		}
		
	}
	else{
		// I am not root, check other stuff
		if(parent && this->parent_index != parent->get_task_index()){
			// Did my parent shift nodes, happens when agent reaches node that is NOT goal
			this->parent_index = parent->get_task_index();
			this->parent_time = parent->get_completion_time();
			this->update_my_travel_time();
		}
		else{
			if(parent && fabs(this->parent_time - parent->get_completion_time()) > 0.2) {
				// did my parent slow down / speed up?
				// has my parents completion time moved? If yes, then I should update my arrival/completion time, probability, and then reward
				this->parent_time = parent->get_completion_time();
				this->last_update_index = update_index;
				this->update_my_completion_time();
			}
		}
	}

	if(this->last_update_index != update_index){
		// I recieved a coordination update, update my probability
		this->last_update_index = update_index;
		this->update_probability_task_is_available();
	}

	// If I don't have kids, make some and roll them out
	if (this->kids.size() == 0) {
		rollout_depth = std::max(0,rollout_depth);
		if (!this->make_kids(task_status, task_set, update_index)) {
			ROS_WARN("Distributed_MCTS::search: Failed to make kids");
			return;
		}
	}
	
	// if I have kids, then select kid with best search reward, and search them
	Distributed_MCTS* gc = NULL;
	if (this->ucb(gc)) {
		// search the kid's branch 
		task_status[gc->get_task_index()] = false; // simulate completing the task
		gc->search(false, depth_in, this, task_status, task_set, rollout_depth, update_index);
		task_status[gc->get_task_index()] = true; // mark the task incomplete, undo simulation

		// do something with the reward
		this->update_down_branch_expected_reward(gc);
	}
	else{
		ROS_WARN("Distributed_MCTS::search: Failed to UCB select kid");
	}
}

void Distributed_MCTS::update_down_branch_expected_reward(Distributed_MCTS* &gc){
	// this checks all th kids to see who has the best expected reward and add it to my reward for my 
	// new best estiamte of my down branch reward. Then, it also updates the expected mean reward for 
	// the kid, gc, who was just searched
    double maxR = -INFINITY;
    double minR = INFINITY;
    int maxI = -1;
    for(int i=0; i<this->kids.size(); i++){
        if(this->kids[i]->down_branch_expected_reward < minR){
            minR = this->kids[i]->down_branch_expected_reward;
        }
        if(this->kids[i]->down_branch_expected_reward > maxR){
            maxR = this->kids[i]->down_branch_expected_reward;
            maxI = i;
        }
    }
    // update my down branch reward
    this->down_branch_expected_reward = this->expected_reward + maxR;
    if(maxI > 0.0){
        // update searched kids cumulative reward
        this->kids[maxI]->cumulative_reward = this->kids[maxI]->cumulative_reward + (this->kids[maxI]->down_branch_expected_reward-minR) / std::max(0.001, maxR-minR);
    }
}

void Distributed_MCTS::update_down_branch_expected_reward() {
	double max_kid_branch_reward = -INFINITY;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_down_branch_expected_reward() > max_kid_branch_reward){
			max_kid_branch_reward = this->kids[i]->get_down_branch_expected_reward();
		}
	}
	this->down_branch_expected_reward = this->expected_reward + max_kid_branch_reward;
}

void Distributed_MCTS::sample_tree(Agent_Coordinator* coord_in, int &depth){
	// Sample the tree and update probable actions

	if(depth == 0){
		// I am the root, reset probable actions
		coord_in->reset_prob_actions();
	}

	// add stop to coordinator
	coord_in->add_stop_to_my_path(this->task_index, this->completion_time, this->branch_probability);
	if(depth > this->max_search_depth){
		// Too deep, don't continue sampling
		return;
	}
   
    // only continue if I have kids
    if(this->kids.size() == 0){
    	return;
    }

    // find max kid!
    double maxR = -INFINITY;
    int maxI = -1;
    for(size_t i=0; i<this->kids.size(); i++){
        if(this->kids[i]->get_down_branch_expected_reward() > maxR){
            maxR = this->kids[i]->get_down_branch_expected_reward();
            maxI = i;
        }
    }
    
    double sumPP = 0.0;
    for(size_t i=0; i<this->kids.size(); i++){
        if(i == maxI){
            this->kids[i]->raw_probability =  this->kids[i]->raw_probability + this->alpha*(1.0 - this->kids[i]->raw_probability);
        }
        else{
            this->kids[i]->raw_probability =  this->kids[i]->raw_probability + this->alpha*(0,0 - this->kids[i]->raw_probability);
        }
        sumPP = sumPP+ this->kids[i]->raw_probability;
    }
    
    depth++;
    for(size_t i=0; i<this->kids.size(); i++){
       this->kids[i]->raw_probability = this->kids[i]->raw_probability/sumPP;  // normalize
       this->kids[i]->branch_probability = this->branch_probability * this->kids[i]->raw_probability;
       if (this->kids[i]->branch_probability > this->min_sampling_probability_threshold){
           this->kids[i]->sample_tree(coord_in, depth);
       }
    }
}

void Distributed_MCTS::sample_tree(int &depth){
	// Sample the tree and update probable actions
	if(depth > this->max_search_depth){
		// Too deep, don't continue sampling
		return;
	}

    // only continue if I have kids
    if(this->kids.size() == 0){
    	return;
    }

    // find max kid!
    double maxR = -INFINITY;
    int maxI = -1;
    for(size_t i=0; i<this->kids.size(); i++){
        if(this->kids[i]->get_down_branch_expected_reward() > maxR){
            maxR = this->kids[i]->get_down_branch_expected_reward();
            maxI = i;
        }
    }
    
    double sumPP = 0.0;
    for(size_t i=0; i<this->kids.size(); i++){
        if(i == maxI){
            this->kids[i]->raw_probability =  this->kids[i]->raw_probability + this->alpha*(1.0 - this->kids[i]->raw_probability);
        }
        else{
            this->kids[i]->raw_probability =  this->kids[i]->raw_probability + this->alpha*(0.0 - this->kids[i]->raw_probability);
        }
        sumPP = sumPP+ this->kids[i]->raw_probability;
    }
    
    depth++; // update depth for next level
    for(size_t i=0; i<this->kids.size(); i++){
       this->kids[i]->raw_probability = this->kids[i]->raw_probability/sumPP;  // normalize
       this->kids[i]->branch_probability = this->branch_probability * this->kids[i]->raw_probability;
       if (this->kids[i]->branch_probability > this->min_sampling_probability_threshold){
           this->kids[i]->sample_tree(depth);
       }
    }
}

void Distributed_MCTS::update_my_travel_time(){
	double dist = 0.0;
	std::vector<int> path;
	if (this->world->a_star(this->task_index, this->parent->get_task_index(), this->agent->get_pay_obstacle_cost(), false, path, dist)) {
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: A* complete");
		this->distance = dist;
		this->travel_time = this->distance / this->agent->get_travel_vel();
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: travel time complete");
		this->completion_time = this->parent_time + this->travel_time + this->work_time;
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: completion time complete");
		this->raw_reward = this->task->get_reward_at_time(this->completion_time);
		this->update_probability_task_is_available();
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got probability");
		this->expected_reward = this->raw_reward * (1.0 - this->probability_task_available);
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got expected_reward");
	}

	else {
		this->expected_reward = -double(INFINITY);
		ROS_WARN("Distributed_MCTS::update_my_travel_time::Bad A* query");
	}

}

void Distributed_MCTS::update_my_completion_time(){
	// new arrival time, already know distance, get reward and the update 
	this->completion_time = this->parent_time + this->work_time + this->travel_time;
	this->raw_reward = this->task->get_reward_at_time(this->completion_time);
	this->update_probability_task_is_available();
}

void Distributed_MCTS::update_probability_task_is_available(){
	// Update probability and everything down stream of it
	double p_taken = 0.0;
	if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
		this->probability_task_available = (1 - p_taken);
		this->expected_reward *= this->probability_task_available;
	}
}

void Distributed_MCTS::get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards){
	ROS_INFO("Distributed_MCTS::get_best_path:node: %i", this->task_index);
	ROS_INFO("Distributed_MCTS::get_best_path: parent_time / travel_time / work_time / completion_time: %0.2f / %0.2f / %0.2f / %0.2f", this->parent_time, this->travel_time, this->work_time, this->completion_time);
	path.push_back(this->task_index);
	times.push_back(this->completion_time);
	rewards.push_back(this->expected_reward);
	if(this->kids.size() > 0){
		int maxI = -1;
		double maxV = -INFINITY;
		for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_INFO("Distributed_MCTS::get_best_path:: kid[%i] has task index %i and branch reward %0.1f", int(i), this->kids[i]->get_task_index(), this->kids[i]->get_down_branch_expected_reward());
			if(this->kids[i]->get_down_branch_expected_reward() > maxV){
				maxV = this->kids[i]->get_down_branch_expected_reward();
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
		if(this->kids[i]->get_down_branch_expected_reward() > maxV){
			maxV = this->kids[i]->get_down_branch_expected_reward();
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
	if(this->kids.size() == 0){
		return;
	}
	for (size_t i = 0; i < this->kids.size(); i++) {
		if (i != max_child) {
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
	}
}

void Distributed_MCTS::burn_branches() {
	if(this->kids.size() == 0){
		return;
	}
	for (size_t i = 0; i < this->kids.size(); i++) {
		this->kids[i]->burn_branches();
		delete this->kids[i];
	}
}
