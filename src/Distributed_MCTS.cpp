#include "Distributed_MCTS.h"
#include "World.h"
#include "Agent.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"
#include "Pose.h"
#include <iostream>
#include "ros/ros.h"


Distributed_MCTS::Distributed_MCTS(World* world, Map_Node* task_in, Agent* agent_in, Distributed_MCTS* parent, const int update_index){
	//ROS_WARN("Distributed_MCTS::Distributed_MCTS: initializing Distributed_MCTS class");
	this->agent = agent_in; // which agent am I?
	this->task = task_in; // which task am I?
	this->task_index = task_in->get_index(); // whats my index?
	this->world = world; // world
	this->use_difference_rewards = true;
	this->sum_n_pulls = 0.0;
	this->probability_task_available = 1.0;
	this->difference_penalty = 0.0;
	this->travel_weight = 0.0;
	

	this->last_update_index = update_index; // when was the last time my coordination info was updated
	
	if (parent) {
		this->parent = parent;
		this->parent_time = parent->get_completion_time();
		this->parent_index = parent->get_task_index();
		this->raw_probability = 0.0; // How likely am I to be selected compared to siblings?
		this->branch_probability = 0.0; // How likely am I to be selected accounting for parent
	}
	else {
		// Set as root if I don't have a parent
		this->raw_probability = 1.0;
		this->branch_probability = 1.0;
		this->parent = NULL;
	}
	

	this->max_search_depth = this->world->search_depth;//6//this->world->get_n_active_tasks();

	// MCTS stuff
	this->raw_reward = 0.0; // How much am I worth at completion time w/o coordination
	this->expected_reward = 0.0; // Including coordination, how much am I worth?
	this->down_branch_expected_reward = 0.0; // the expected reward for my best kids and their kids (recursive to max depth)
	this->number_pulls = 0; // how many times have I been pulled
	this->max_rollout_depth = 0; //edit these to ensure tree can grow on simple case


	// sampling stuff
	this->alpha = this->world->alpha; //0.01;// Gradient ascent rate for coordination / learning rate on probable actions
	this->min_sampling_probability_threshold = this->world->min_sampling_threshold; //0.005; // How far down the tree will I search?

	// D-UCB Stuff
	this->cumulative_reward = 0.0; // what is the total cumulative reward across all historical pulls
	this->mean_reward = 0.0; // cumulative reward / number_pulls
	this->gamma = this->world->gamma; //0.999; // How long of a memory should I keep?
	this->epsilon = this->world->epsilon; //0.5; // UCB weight term for parent number_pulls
	this->beta = this->world->beta; //1.41; // explore vs exploit

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
			this->raw_reward = this->world->get_task_reward_at_time(this->task_index, this->completion_time);
			//ROS_INFO("Distributed_MCTS::Distributed_MCTS: got raw_reward: %0.2f", this->raw_reward);
			this->update_probability_task_is_available();
			this->get_difference_penalty();
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got probability_task_available: %0.2f", this->probability_task_available);
			this->expected_reward = this->raw_reward * this->probability_task_available - this->difference_penalty - this->travel_time * this->travel_weight;
			//ROS_INFO("Distributed_MCTS::Distributed_MCTS: got expected_reward: %0.2f", this->expected_reward);
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got expected_reward");
			this->down_branch_expected_reward = this->expected_reward;
			this->number_pulls = 1; // This kinda counts right?
		}

		else {
			this->expected_reward = 0.0;
			this->down_branch_expected_reward = this->expected_reward;
			//ROS_WARN("Distributed_MCTS::Initialize::Bad A* query");
		}
	}
	else{
		if (task_in->is_active()) {
			// this is the root! I am already here!!!
			this->parent_time = this->world->get_c_time();
			this->agent->get_travel_time(this->task_index, this->travel_time);
			this->work_time = task_in->get_time_to_complete(this->agent, world);
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			this->raw_reward = this->world->get_task_reward_at_time(this->task_index, this->completion_time);
			//ROS_INFO("Distributed_MCTS::Distributed_MCTS: got raw_reward: %0.2f", this->raw_reward);
			this->update_probability_task_is_available();
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got probability_task_available: %0.2f", this->probability_task_available);
			this->get_difference_penalty();
			this->expected_reward = this->raw_reward * this->probability_task_available - this->difference_penalty - this->travel_time * this->travel_weight;
			//ROS_INFO("Distributed_MCTS::Distributed_MCTS: got expected_reward: %0.2f", this->expected_reward);
			//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got expected_reward");
			this->down_branch_expected_reward = this->expected_reward;
		}
		else{
			this->parent_time = this->world->get_c_time();
			this->agent->get_travel_time(this->task_index, this->travel_time);
			this->work_time = 0.0;
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			this->raw_reward = 0.0;
			this->expected_reward = 0.0;
			this->down_branch_expected_reward = 0.0;
		}
	}
}

void Distributed_MCTS::get_difference_penalty(){
	this->difference_penalty = 0.0;
	if(this->use_difference_rewards){
		std::vector<double> probs;
		std::vector<double> times;
		// Check if there are any claimed after me that could affect the difference penalty
		this->world->get_claims_after(this->task_index, this->completion_time, this->agent->get_index(), probs, times);
		if(probs.size() > 0){
			double prior_prob = 1.0 - this->probability_task_available;
			for(size_t i=0; i<times.size(); i++){
				this->difference_penalty += (probs[i]-prior_prob) * std::max(this->world->get_task_reward_at_time(this->task_index, times[i]), 0.0);
				prior_prob = probs[i];
			}
		}
	}
}

void Distributed_MCTS::clean_task(const int &ni){
	int i=0;
	while(i<this->kids.size()){
		if(this->kids[i]->get_task_index() == ni ){
			this->kids[i]->burn_branches();
			delete this->kids[i];
			this->kids.erase(this->kids.begin() + i);
		}
		else{
			this->kids[i]->clean_task(ni);
			i++;
		}
	}
}

Distributed_MCTS::~Distributed_MCTS(){
	// DO NOT BURN BRANCHES!!! This is taken care of in Agent Planning destructor and will cause a double delete when called as it is already handled in burn branches, recalling just does it twice (which gives an error)
}

bool Distributed_MCTS::make_kids( const std::vector<bool> &task_status, const std::vector<int> &task_set, const int &update_index ) {
	// Make kids
	// task_set is a list of all available tasks
	// task status is a list of tasks current status
	// update index is when the coordination was last checked

	//ROS_WARN("Distributed_MCTS::make_kids: in with task_set.size(): %i", int(task_set.size()));
	bool kids_made = false;
	for (size_t j = 0; j < task_set.size(); j++) { // potentially add a kid for each active task
		//ROS_WARN("Distributed_MCTS::make_kids: checking kid[%i]", int(j));
		int ti = task_set[j]; // which task am I looking at
		if (task_status[ti]) { // if task ti needs to be completed
			//ROS_WARN("Distributed_MCTS::make_kids[%i]: task[%i] is active", this->task_index, int(ti));
			Distributed_MCTS* kiddo = new Distributed_MCTS(this->world, this->world->get_nodes()[ti], this->agent, this, update_index);
			this->kids.push_back(kiddo);
			kids_made = true;
		}
	}

	if(kids_made){
		this->update_down_branch_expected_reward();
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

bool Distributed_MCTS::ucb(int &gc){
    //exactly the same as other ucb, but uses the historical mean value
    //of the cumulative reward to pick child. Works with D-UCB by adding a Gamma
    //ROS_ERROR("in ucb %i", this->task_index);
    gc = -1;
    if(this->kids.size() == 0){
    	return false;
    }

    double minR = INFINITY;
    double maxR = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        this->kids[i]->cumulative_reward = this->kids[i]->cumulative_reward * this->gamma;
        this->kids[i]->mean_reward = this->kids[i]->cumulative_reward / std::max(0.01,this->kids[i]->number_pulls);
        //ROS_INFO("Distributed_MCTS::ucb: [%i]->kids[%i/%i]: down_branch_expected_reward %0.2f/ cumulative_reward: %0.2f/ mean_reward: %0.2f/ number_pulls %0.2f", this->task_index, i, this->kids[i]->get_task_index(), this->kids[i]->get_down_branch_expected_reward(), this->kids[i]->get_cumulative_reward(), this->kids[i]->mean_reward, std::max(0.01,this->kids[i]->number_pulls));
        if(this->kids[i]->mean_reward < minR){
            minR = this->kids[i]->mean_reward;
        }
        if(this->kids[i]->mean_reward > maxR){
            maxR = this->kids[i]->mean_reward;
        }
    }
   // ROS_WARN("Distributed_MCTS::ucb: maxR %0.2f and minR %0.2f", maxR, minR);

    double maxU = -INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        int task_index = this->kids[i]->get_task_index();
    	double rr = (this->kids[i]->mean_reward-minR) / std::max(0.01,(maxR-minR));
        //ROS_INFO("Distributed_MCTS::ucb: kids[%i] has mean_reward: %0.2f with maxR %0.2f and minR %0.2f", i, this->kids[i]->mean_reward, maxR, minR);
        this->kids[i]->number_pulls = this->kids[i]->number_pulls * this->kids[i]->gamma;
        double ee = this->kids[i]->beta*sqrt(this->kids[i]->epsilon*log(std::max(0.01,this->number_pulls))/std::max(0.01,this->kids[i]->number_pulls));
        //ROS_INFO("Distributed_MCTS::ucb: kids[%i] has rr %0.2f + ee %0.2f = %0.2f against maxM %0.2f", int(i), rr,ee,rr+ee, maxM);
        if (rr + ee > maxU){
        	//ROS_INFO("Distributed_MCTS::ucb: new max kid");
            maxU = rr + ee;
            gc = i;
        }
    }
    return true;
}

void Distributed_MCTS::search(int depth_in, Distributed_MCTS* parent, std::vector<bool> &task_status, std::vector<int> &task_set, int rollout_depth, const int &update_index) {
	//ROS_ERROR("into search at depth: %i", depth_in);
	//ROS_INFO("Distributed_MCTS::Search: Searching: %i at depth %i and time %0.2f", this->task_index, depth_in, this->completion_time);
	this->number_pulls++;
	this->sum_n_pulls++;
	if (task_status[this->task_index] == true && this->parent) {
		ROS_ERROR("Distributed_MCTS::search: bad task selected %i at depth %i and rollout_depth %i", this->task_index, depth_in, rollout_depth);
		this->expected_reward = 0.0;
		this->down_branch_expected_reward = 0.0;
		return;
	}

	if(this->parent != parent){
		ROS_ERROR("Parent Error %i at depth %i", this->task_index, depth_in);
	}
	
	depth_in++;
	
	if(rollout_depth >= 0){
		// I am part of the rollout, increase the depth
		rollout_depth++;
	}

	if (this->parent && (depth_in > this->max_search_depth || parent->get_completion_time() > this->world->get_end_time() || rollout_depth > this->max_rollout_depth)) {
		return;
	}
	//ROS_ERROR("into search: going into check_completion_time");
	 // Since I am moving, check and make sure that I am still being completed at the right time
	this->check_completion_time(depth_in, update_index);

	//ROS_ERROR("into search: checking if I should update goal");
	if(this->last_update_index != update_index){
		// I recieved a coordination update, update my probability
		this->last_update_index = update_index;
		this->clean_kids();
		this->update_probability_task_is_available();
		this->get_difference_penalty();
		this->expected_reward = this->raw_reward * this->probability_task_available - this->difference_penalty -  this->travel_time * this->travel_weight;
	}

	//ROS_ERROR("into search: checking for kids");
	// If I don't have kids, make some and roll them out, already added +1 to depth_in
	if (this->kids.size() == 0 && depth_in < this->max_search_depth ) {
		rollout_depth = std::max(0,rollout_depth); // If first time making kids then start rollout counter, if not then continue incrementing
		if (!this->make_kids(task_status, task_set, update_index)) {
			//ROS_WARN("Distributed_MCTS::search: Failed to make kids");
			return;
		}
	}
	
	//ROS_ERROR("into search: going into ucb");
	if(this->kids.size()){
		// if I have kids, then select kid with best search reward, and search them
		int gc = -1;
		if(this->ucb(gc)){
			// search the kid's branch 
			task_status[this->kids[gc]->get_task_index()] = false; // simulate completing the task
			this->kids[gc]->search(depth_in, this, task_status, task_set, rollout_depth, update_index);
			task_status[this->kids[gc]->get_task_index()] = true; // mark the task incomplete, undo simulation

			// do something with the reward
			this->update_down_branch_expected_reward(gc);
		}
		else{
			ROS_WARN("Distributed_MCTS::search: Failed to UCB select kid");
		}
	}
}

void Distributed_MCTS::clean_kids(){
	for(size_t i =0; i<this->kids.size(); i++){
		int ti = this->kids[i]->get_task_index();
		if(!this->world->get_task_status(ti)){
			this->kids[i]->burn_branches();
			delete this->kids[i];
			this->kids.erase(this->kids.begin() + i);
			i--;
		}
	}
}

void Distributed_MCTS::update_down_branch_expected_reward(const int &gc){
	// this checks all th kids to see who has the best expected reward and add it to my reward for my 
	// new best estiamte of my down branch reward. Then, it also updates the expected mean reward for 
	// the kid, gc, who was just searched
    double maxR = -INFINITY;
    double minR = INFINITY;
    for(int i=0; i<this->kids.size(); i++){
        if(this->kids[i]->down_branch_expected_reward < minR){
            minR = this->kids[i]->get_down_branch_expected_reward();
        }
        if(this->kids[i]->down_branch_expected_reward > maxR){
            maxR = this->kids[i]->get_down_branch_expected_reward();
        }
	    //ROS_INFO("Distributed_MCTS::update_down_branch_expected_reward: down_branch_expected_reward[%i] kid[%i/%i]: %0.2f", this->task_index, int(i), this->kids[i]->get_task_index(), this->kids[i]->get_down_branch_expected_reward());
    }
    //ROS_INFO("Distributed_MCTS::update_down_branch_expected_reward: [%i] my down_branch_expected_reward: %0.2f", this->task_index, this->down_branch_expected_reward);
    // update my down branch reward
    this->down_branch_expected_reward = this->expected_reward + maxR;
    //ROS_INFO("Distributed_MCTS::update_down_branch_expected_reward: [%i] my down_branch_expected_reward: %0.2f", this->task_index, this->down_branch_expected_reward);

    // update searched kids cumulative reward
    //ROS_INFO("Distributed_MCTS::update_down_branch_expected_reward: gc=kids[%i]->cumulative_reward: %0.2f with %0.2f and %0.2f", gc->get_task_index(), gc->get_cumulative_reward(), minR, maxR);
    this->kids[gc]->set_cumulative_reward( this->kids[gc]->get_cumulative_reward() + (this->kids[gc]->get_down_branch_expected_reward()-minR) / std::max(0.001, maxR-minR));
    //ROS_INFO("Distributed_MCTS::update_down_branch_expected_reward: gc=kids[%i]->cumulative_reward: %0.2f with %0.2f and %0.2f", gc->get_task_index(), gc->get_cumulative_reward(), minR, maxR);
}


void Distributed_MCTS::update_down_branch_expected_reward() {

    double maxR = -INFINITY;
    double minR = INFINITY;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_down_branch_expected_reward() < minR){
            minR = this->kids[i]->get_down_branch_expected_reward();
        }
        if(this->kids[i]->get_down_branch_expected_reward() > maxR){
            maxR = this->kids[i]->get_down_branch_expected_reward();
        }
	}
	this->down_branch_expected_reward = this->expected_reward + maxR;

	// Now update the cumulative reward of each kid, this is weighted 0->1 and by gamma
	for(size_t i=0; i<this->kids.size(); i++){
		this->kids[i]->set_cumulative_reward(this->kids[i]->get_cumulative_reward() + (this->kids[i]->get_down_branch_expected_reward()-minR) / std::max(0.001, maxR-minR));
	}
}


void Distributed_MCTS::check_completion_time(const int &depth, const int &update_index){
	// Ensure that the time I am broadcasting includes the most up to date information!
	//ROS_WARN("Distributed_MCTS::check_completion_time:%i at depth %i and time %0.2f", this->task_index, depth, this->completion_time);
	
	if (!this->parent && this->last_update_index != update_index){
		// I am the root and this is the first search of this iteration, check my travel time and update completion time
		//ROS_ERROR("Distributed_MCTS::check_completion_time: getting completion time, currently it is: %0.2f", this->completion_time);
		//ROS_ERROR("Distributed_MCTS::check_completion_time: 	for goal: %i", this->task_index);
		if(this->agent->get_travel_time(this->task_index, this->travel_time)){
			this->parent_time = this->world->get_c_time();
			this->completion_time = this->parent_time + this->travel_time + this->work_time;
			//ROS_ERROR("Distributed_MCTS::check_completion_time: got completion time: %0.2f = parent_time %0.2f + travel_time %0.2f + work_time %0.2f", this->completion_time, this->parent_time, this->travel_time, this->work_time);
		}
		
	}
	else{
		// I am not root, check other stuff
		if(this->parent && this->parent_index != this->parent->get_task_index()){
			// Did my parent shift nodes, happens when agent reaches node that is NOT goal
			//ROS_ERROR("Distributed_MCTS::check_completion_time: getting travel time, currently completion_time is: %0.2f", this->completion_time);
			this->parent_index = this->parent->get_task_index();
			this->parent_time = this->parent->get_completion_time();
			this->update_my_travel_time();
			//ROS_ERROR("Distributed_MCTS::check_completion_time: got travel_time time, currently completion_time is: %0.2f", this->completion_time);
		}
		else{
			if(this->parent && fabs(this->parent_time - this->parent->get_completion_time()) > 0.2) {
				//ROS_ERROR("Distributed_MCTS::check_completion_time: getting completion time, currently it is: %0.2f", this->completion_time);
				// did my parent slow down / speed up?
				// has my parents completion time moved? If yes, then I should update my arrival/completion time, probability, and then reward
				//ROS_WARN("Distributed_MCTS::check_completion_time:Updating parent time from %0.2f giving completion_time %0.2f", this->parent_time, this->completion_time);
				this->parent_time = this->parent->get_completion_time();
				this->last_update_index = update_index;
				this->update_my_completion_time();
				//ROS_ERROR("Distributed_MCTS::check_completion_time: got completion time, currently it is: %0.2f", this->completion_time);
				//ROS_WARN("Distributed_MCTS::sample_tree:Updated parent time to %0.2f giving completion_time %0.2f", this->parent_time, this->completion_time);
			}
		}
	}
	//ROS_WARN("Distributed_MCTS::check_completion_time:%i at depth %i and time %0.2f", this->task_index, depth, this->completion_time);
}

void Distributed_MCTS::sample_tree(Agent_Coordinator* coord_in, int depth, const int &update_index){
	// Sample the tree and update probable actions

	if(depth == 0){
		// I am the root, reset probable actions
		coord_in->reset_prob_actions();
		this->branch_probability = 1.0;
	}

	this->check_completion_time(depth, update_index);

	// add stop to coordinator
	//ROS_INFO("Distributed_MCTS::sample_tree: [%i]", this->task_index);
	coord_in->add_stop_to_my_path(this->task_index, this->completion_time, this->branch_probability);
	if(depth > this->max_search_depth){
		//ROS_INFO("Distributed_MCTS::sample_tree: returning because depth %i is greater than max depth %i", depth, this->max_search_depth);
		// Too deep, don't continue sampling
		return;
	}
	else{		
	    depth++;
	}
   
    // only continue if I have kids
    if(this->kids.size() == 0){
    	//ROS_INFO("Distributed_MCTS::sample_tree: returning becasue kids=0");
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
    //ROS_INFO("Distributed_MCTS::sample_tree:maxR %0.2f and maxI %i", maxR, maxI);
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
    //ROS_INFO("Distributed_MCTS::sample_tree:sumPP: %0.2f", sumPP);
    
    for(size_t i=0; i<this->kids.size(); i++){
       this->kids[i]->raw_probability = this->kids[i]->raw_probability/sumPP;  // normalize
       //ROS_INFO("Distributed_MCTS::sample_tree:kids[%i] raw_probability %0.2f", int(i), this->kids[i]->raw_probability);
       this->kids[i]->branch_probability = this->branch_probability * this->kids[i]->raw_probability;
       //ROS_INFO("Distributed_MCTS::sample_tree:branch_probability %0.2f", this->branch_probability);
       //ROS_INFO("Distributed_MCTS::sample_tree: kids[%i] branch_probability %0.2f and min_sampling_probability_threshold %0.2f", int(i), this->kids[i]->branch_probability, this->min_sampling_probability_threshold);
       if (this->kids[i]->branch_probability > this->min_sampling_probability_threshold){
           this->kids[i]->sample_tree(coord_in, depth, update_index);
       }
    }
}

void Distributed_MCTS::sample_tree(int depth){
	// Sample the tree and update probable actions
		if(depth == 0){
		// I am the root, reset probable actions
		this->branch_probability = 1.0;
	}

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
		this->raw_reward = this->world->get_task_reward_at_time(this->task_index, this->completion_time);
		this->update_probability_task_is_available();
		this->get_difference_penalty();
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got probability");
		this->expected_reward = this->raw_reward * this->probability_task_available - this->difference_penalty - this->travel_time * this->travel_weight;
		//ROS_WARN("Distributed_MCTS::Distributed_MCTS: got expected_reward");
	}

	else {
		this->expected_reward = 0.0;
		ROS_WARN("Distributed_MCTS::update_my_travel_time::Bad A* query");
	}

}

void Distributed_MCTS::update_my_completion_time(){
	// new arrival time, already know distance, get reward and the update 
	this->completion_time = this->parent_time + this->work_time + this->travel_time;
	this->raw_reward = this->world->get_task_reward_at_time(this->task_index, this->completion_time);
	this->update_probability_task_is_available();
	this->get_difference_penalty();
	this->expected_reward = this->raw_reward * this->probability_task_available - this->difference_penalty - this->travel_time - this->travel_weight;
}

void Distributed_MCTS::update_probability_task_is_available(){
	// Update probability and everything down stream of it
	double p_taken = 0.0;
	if (this->agent->get_coordinator()->get_advertised_task_claim_probability(this->task_index, this->completion_time, p_taken, this->world)) {
		this->probability_task_available = (1 - p_taken);
	}
}

void Distributed_MCTS::get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards, int depth, const int &update_index){

	this->check_completion_time(depth, update_index);

	//ROS_INFO("Distributed_MCTS::get_best_path at node: %i: with parent_time / travel_time / work_time / completion_time: %0.2f / %0.2f / %0.2f / %0.2f", this->task_index, this->parent_time, this->travel_time, this->work_time, this->completion_time);
	if(this->world->get_task_status(this->task_index)){
		path.push_back(this->task_index);
		times.push_back(this->completion_time);
		rewards.push_back(this->expected_reward);
	}
	if(this->kids.size() > 0){
		int maxI = -1;
		double maxV = 0.01;
		for(size_t i=0; i<this->kids.size(); i++){
		//	ROS_INFO("Distributed_MCTS::get_best_path:: kid[%i] has task index %i and branch reward %0.1f", int(i), this->kids[i]->get_task_index(), this->kids[i]->get_down_branch_expected_reward());
			if(this->kids[i]->get_down_branch_expected_reward() > maxV){
				maxV = this->kids[i]->get_down_branch_expected_reward();
				maxI = i;
			}
		}
		//ROS_WARN("Distributed_MCTS::get_best_path:: maxI %i and maxV %0.1f for task %i", maxI, maxV, this->kids[maxI]->get_task_index());
		if(maxI >= 0){
			this->kids[maxI]->get_best_path(path, times, rewards, depth, update_index);
		}
	}
}

bool Distributed_MCTS::exploit_tree(int &max_kid_index, std::vector<std::string> &args, std::vector<double> &vals, int depth, const int &update_index) {
	this->check_completion_time(depth, update_index);

	double maxR = -INFINITY;
	int maxI = -1;
	for(size_t i=0; i<this->kids.size(); i++){
		if(this->kids[i]->get_down_branch_expected_reward() > maxR){
			maxR = this->kids[i]->get_down_branch_expected_reward();
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
	if(this->kids.size() <= 1){
		return;
	}

	for (size_t i = 0; i < this->kids.size(); i++) {
		if (i != max_child) {
			this->kids[i]->burn_branches();
			delete this->kids[i];
		}
	}
	
	Distributed_MCTS* gc = this->kids[max_child];
	this->kids.clear();
	this->kids.push_back(gc);
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
