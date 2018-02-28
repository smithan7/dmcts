#include "Agent_Planning.h"
#include "Agent_Coordinator.h"
#include "Map_Node.h"
#include "Agent.h"
#include "World.h"
#include "MCTS.h"
#include "D_MCTS.h"
#include "Distributed_MCTS.h"
#include "Goal.h"
#include "Pose.h"

#include <iostream>
#include <fstream>

#include <ros/ros.h>

Agent_Planning::Agent_Planning(Agent* agent, World* world_in){
	this->agent = agent;
	this->world = world_in;
	this->task_selection_method = agent->get_task_selection_method();
	this->planning_iter = 0;
	this->last_planning_iter_end = -1;
	this->initial_search_time = 5.0;//this->agent->plan_duration.toSec();
	this->reoccuring_search_time = 0.95 * this->agent->plan_duration.toSec();
	this->coord_update = 0;
}

void Agent_Planning::Distributed_MCTS_task_by_completion_reward() {
	ROS_WARN("Agent_Planning::Distributed_MCTS_task_by_completion_reward: in 'Distributed_MCTS_task_by_completion_reward' on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	double reward_in = 0.0;
	double s_time = double(clock()) / double(CLOCKS_PER_SEC);
	std::vector<bool> task_list; // list of all tasks status, true if active, false if complete
	std::vector<int> task_set; // list, by index, of all active tasks
	this->world->get_task_status_list(task_list, task_set);
	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: got task_list (%i) / task_set (%i)", int(task_list.size()), int(task_set.size()));
	if (!this->dist_mcts) {
		//ROS_WARN("Agent_Planning::Distributed_MCTS_task_by_completion_reward: initializing dist_mcts");
		this->dist_mcts = new Distributed_MCTS(this->world, this->world->get_nodes()[this->get_agent()->get_loc()], this->get_agent(), NULL, this->coord_update);
	}
	this->coord_update++; // Make it a new planning iteration
	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: finished initializing D_MCTS");

	//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: going into search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	// task of root is marked complete
	//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: set task index (%i) to false", this->dist_mcts->get_task_index());
	
	task_list[this->dist_mcts->get_task_index()] = false;
	int depth_in = 0;
	Distributed_MCTS* parent_of_none = NULL; // this gets set in Dist-MCTS Root
	int rollout_depth = -1; // Indicate rollout has NOT started!
	int planning_iter = 0;
	ROS_WARN("Agent_Planning::Distributed_MCTS_task_by_completion_reward: Have dist_mcts root: %i",this->dist_mcts->get_task_index());
	while( double(clock()) / double(CLOCKS_PER_SEC) - s_time <= this->reoccuring_search_time){
		planning_iter++;
		//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: really going into search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
		//2-I am still not updating the path properly, not rebasing when reaching a new (non-goal) node
		this->dist_mcts->search(true, depth_in, parent_of_none, task_list, task_set, rollout_depth, coord_update);
		if( planning_iter % 1 == 0){
			//this->dist_mcts->sample_tree(depth_in);
		}
		//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: out of search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	}

	ROS_INFO("Agent_Planning::D_MCTS_task_selection: finished searching tree for %i inters", planning_iter);
	this->cumulative_planning_iters += planning_iter;

	this->agent->get_coordinator()->reset_prob_actions(); // clear out probable actions before adding the new ones
	this->dist_mcts->sample_tree(this->agent->get_coordinator(), depth_in, coord_update);
	//printf("sampling time: %0.2f \n", double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	std::vector<int> best_path;
	std::vector<double> times;
	std::vector<double> rewards;
	this->dist_mcts->get_best_path(best_path, times, rewards, depth_in, coord_update);

	std::cout << "Agent_Planning::Distributed_MCTS_task_by_completion_reward:[" << this->agent->get_index() << "]: best_path: ";
	for(size_t i=0; i<best_path.size(); i++){
		//std::cout << std::fixed << std::setprecision(2) << best_path[i] << ", ";
		std::cout << std::fixed << std::setprecision(2) << " ( Path[" << i << "]: " << best_path[i] << " @ " << times[i] << " for " << rewards[i] <<"), ";// << " with probs: " << probs[i] << "), ";
	}
	std::cout << std::endl;
	this->agent->set_path(best_path);
	

	
	//std::vector<double> probs(int(best_path.size()), 1.0);
	//this->agent->get_coordinator()->upload_new_plan(best_path, times, probs);
	//std::cout << "Agent_Planning::Distributed_MCTS_task_by_completion_reward:[" << this->agent->get_index() << "]: best_path: ";
	//for(size_t i=0; i<best_path.size(); i++){
	//	std::cout << std::fixed << std::setprecision(2) << " ( Path[" << i << "]: " << best_path[i] << " @ " << times[i] << " for " << rewards[i] <<"), ";// << " with probs: " << probs[i] << "), ";
	//}
	//std::cout << std::endl;
	
	/*
	std::ofstream outfile;
	outfile.open("planning_time.txt", std::ios::app);
	char buffer[50];
	int n = sprintf_s(buffer, "%i, %0.6f\n", planning_iters, double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	outfile << buffer;
	outfile.close();
	*/

	
	ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward::Agent[%i]'s coord tree", this->agent->get_index());
	this->agent->get_coordinator()->print_prob_actions();
	for(int i=0; i<this->world->get_n_agents(); i++){
		if(i != this->agent->get_index()){
			ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward::Agent[%i]'s understanding of agent[%i]s coord tree", this->agent->get_index(), i);
			this->world->get_agents()[i]->get_coordinator()->print_prob_actions();
		}
	}
	
	if(this->world->get_c_time() > this->initial_search_time && this->agent->get_at_node()){ // Don't explot path / settle on waht to do until I've had a few seconds to think and I am at a node
		// I am at either edge.x / edge.y
		int max_kid_index = -1;
		std::vector<std::string> args;
		std::vector<double> vals;
		//ROS_ERROR("Agent_Planning::D_MCTS_task_selection: I am at a node and I am on edge: %i/%i with goal: %i and mcts root %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index(), this->dist_mcts->get_task_index());
		if (this->dist_mcts->exploit_tree(max_kid_index, args, vals, depth_in, coord_update)) {
			int goal_task_index = this->dist_mcts->get_kids()[max_kid_index]->get_task_index();
			//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: exploit_tree succesful with max_kid_index %i and task index %i", max_kid_index, goal_task_index);
			// only make a new goal if the current goai is NOT active
			//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: checking if current goal (%i) is active, it is %i", this->agent->get_goal()->get_index(), this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active());
	
			if(this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active()){
				// My current goal is still active, check if I am at my goal
				if(this->agent->get_edge().x == this->agent->get_goal()->get_index()){
					// I am still at my goal and it is not complete, continue
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: I am at my goal %i and it is not complete, returning", this->agent->get_goal()->get_index());
					return;
				}
				else{
					// I am NOT at my goal and it is still active
					// THis is where I allow the agent to change goals
					// My goal is not active, set my new goal
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: I am at node: %i/%i with an active old goal: %i, setting new goal %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index(), goal_task_index);
					if(this->dist_mcts->get_task_index() == this->agent->get_goal()->get_index()){
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: Goal == Dist-MCTS root, returning");
						return;
					}
					// Am I nbrs with my goal task
					if (world->are_nbrs(this->agent->get_loc(), goal_task_index) ) {
						// I am nbrs with my goal task, assign it as my goal

						this->set_goal(goal_task_index);
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: %i is nbrs with %i", this->agent->get_edge().x, goal_task_index);
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: pruning dist_mcts branches except max_kid_index: %i", goal_task_index);
						this->dist_mcts->prune_branches(max_kid_index);
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: pruned branches and now have kids: %i", int(this->dist_mcts->get_kids().size()));
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: resetting dist_mcts root to %i", goal_task_index);
						Distributed_MCTS* old = this->dist_mcts;
						this->dist_mcts = this->dist_mcts->get_kids()[max_kid_index];
						//delete old;
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: dist_mcts has task_index %i and %i kids", this->dist_mcts->get_task_index(), int(this->dist_mcts->get_kids().size()));
						this->dist_mcts->set_as_root();
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: reset dist_mcts root");
					}
					else{
						// I am not nbrs with my goal task, assign a temporary task to dist-mcts root and assign goal to robot
						std::vector<int> path;
						this->set_goal(this->agent->get_goal()->get_index(), path);
						/*** if i am switching to a temporary task, rebase the root as myself, however, do not prune kids. ***/			
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: %i is NOT nbrs with %i", this->agent->get_edge().x, goal_task_index);
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: set new max_kid_index: %i", goal_task_index);
						
						// I am not nbrs with the next node (my goal), replace root index with current node but don't advance/prune tree
						this->dist_mcts->prune_branches(max_kid_index);
						//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: got path");
						//std::cout << "    ";
						//for(size_t i=0; i<path.size(); i++){
						//	std::cout << path[i] << ",";
						//}
						//std::cout << std::endl;
						if(path.size()>2){
							this->dist_mcts->set_task_index(path[1]);
							//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: set task index to %i", path[1]);
						}
					}
				}
			}
			else{
				// My goal is not active, set my new goal
				//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: I am at node: %i/%i with A COMPLETED old goal: %i, setting new goal %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index(), goal_task_index);
				
	
				// Am I nbrs with my goal task
				if (world->are_nbrs(this->agent->get_loc(), goal_task_index) ) {
					// I am nbrs with my goal task, assign it as my goal
					this->set_goal(goal_task_index);
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: %i is nbrs with %i", this->agent->get_edge().x, goal_task_index);
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: pruning dist_mcts branches except max_kid_index: %i", goal_task_index);
					this->dist_mcts->prune_branches(max_kid_index);
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: pruned branches and now have kids: %i", int(this->dist_mcts->get_kids().size()));
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: resetting dist_mcts root to %i", goal_task_index);
					Distributed_MCTS* old = this->dist_mcts;
					this->dist_mcts = this->dist_mcts->get_kids()[0];
					//delete old;
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: dist_mcts has task_index %i and %i kids", this->dist_mcts->get_task_index(), int(this->dist_mcts->get_kids().size()));
					this->dist_mcts->set_as_root();
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: reset dist_mcts root");
				}
				else{
					// I am not nbrs with my goal task, assign a temporary task
					std::vector<int> path;
					this->set_goal(goal_task_index, path);
					/*** if i am switching to a temporary task, rebase the root as myself, however, do not prune kids. ***/			
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: %i is NOT nbrs with %i", this->agent->get_edge().x, goal_task_index);
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: set new max_kid_index: %i", goal_task_index);
					
					// I am not nbrs with the next node (my goal), replace root index with current node but don't advance/prune tree
					this->dist_mcts->prune_branches(max_kid_index);
					//ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: got path");
					std::cout << "    ";
					for(size_t i=0; i<path.size(); i++){
						std::cout << path[i] << ",";
					}
					std::cout << std::endl;
					if(path.size()>2){
						this->dist_mcts->set_task_index(path[1]);
					//	ROS_INFO("Agent_Planning::Distributed_MCTS_task_by_completion_reward: set task index to %i", path[1]);
					}
				}
			}
		}
	}
	//ROS_WARN("Agent_Planning::D_MCTS_task_selection: edge out %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
}



Agent_Planning::~Agent_Planning(){}

void Agent_Planning::plan() {
	//std::cerr << "Agent_Planning::plan::this->task_selection_method: " << this->task_selection_method << std::endl;

	// randomly select nbr node
	if (this->task_selection_method.compare("random_nbr") == 0) {
		this->select_random_nbr();
	}

	// randomly select a node on the graph
	else if (this->task_selection_method.compare("random_node") == 0) {
		this->select_random_node();
	}

	// randomly select a node on the graph
	else if (this->task_selection_method.compare("random_task") == 0) {
		this->select_random_task();
	}

	// greedily select a task by current reward
	else if (this->task_selection_method.compare("greedy_current_reward") == 0) {
		this->select_greedy_task_by_current_reward();
	}

	// greedily select a task by current reward
	else if (this->task_selection_method.compare("greedy_arrival_reward") == 0) {
		this->select_greedy_task_by_arrival_reward();
	}

	// greedily select a task by current reward
	else if (this->task_selection_method.compare("greedy_completion_reward") == 0) {
		//ROS_ERROR("Agent_Planning[%i]::plan: in to select_greedy_task_by_completion_reward", this->agent->get_index());
		this->select_greedy_task_by_completion_reward();
		//ROS_ERROR("Agent_Planning[%i]::plan: out of select_greedy_task_by_completion_reward", this->agent->get_index());
	}

	// greedily select task by arrival time
	else if (this->task_selection_method.compare("greedy_arrival_time") == 0) {
		this->select_greedy_task_by_arrival_time();
	}

	// greedily select task by completion time
	else if (this->task_selection_method.compare("greedy_completion_time") == 0) {
		this->select_greedy_task_by_completion_time();
	}

	// select task by current value
	else if (this->task_selection_method.compare("value_current") == 0) {
		this->select_task_by_current_value();
	}

	// select task by value at arrival time
	else if (this->task_selection_method.compare("value_arrival") == 0) {
		this->select_task_by_arrival_value();
	}

	// select task by value at completion time
	else if (this->task_selection_method.compare("value_completion") == 0) {
		this->select_task_by_completion_value();
	}

	// // select task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete
	else if (this->task_selection_method.compare("impact_completion_reward") == 0) {
		this->select_task_by_impact_completion_reward();
	}

	// select task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete - (travel_time + work_time)
	else if (this->task_selection_method.compare("impact_completion_value") == 0) {
		this->select_task_by_impact_completion_value();
	}
	// select task by MCTS using reward at time of completion
	else if (this->task_selection_method.compare("mcts_task_by_completion_reward") == 0) {
		this->world->set_mcts_reward_type("normal");
		this->MCTS_task_by_completion_reward();
	}
	// select task by MCTS using reward at time of completion and gradient descent on method
	else if (this->task_selection_method.compare("mcts_task_by_completion_reward_gradient") == 0) {
		this->world->set_mcts_reward_type("normal");
		//ROS_INFO("Agent_Planning::plan: going into D_MCTS_task_by_completion_reward on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
		//this->D_MCTS_task_by_completion_reward();
		this->Distributed_MCTS_task_by_completion_reward();
	}
	// select task by MCTS using value at time of completion
	else if (this->task_selection_method.compare("mcts_task_by_completion_value") == 0) {
		this->world->set_mcts_reward_type("normal");
		this->MCTS_task_by_completion_value();
	}
	// select task by MCTS using reward impact at time of completion
	else if (this->task_selection_method.compare("mcts_task_by_completion_reward_impact") == 0) {
		//ROS_INFO("Agent_Planning::plan: into 'mcts_task_by_completion_reward_impact'");
		this->world->set_mcts_reward_type("impact");
		this->MCTS_task_by_completion_reward_impact();
		//ROS_INFO("Agent_Planning::plan: out of 'mcts_task_by_completion_reward_impact'");
	}
	// select task by MCTS using value impact at time of completion
	else if (this->task_selection_method.compare("mcts_task_by_completion_value_impact") == 0) {
		this->world->set_mcts_reward_type("impact");
		this->MCTS_task_by_completion_value_impact();
	}
	// method was not found, let user know
	else {
		ROS_ERROR("Agent_Planning::plan:: goal finding method unspecified");
	}
}

void Agent_Planning::select_task_by_impact_completion_reward() {
	// select task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete
	Agent_Coordinator* my_coord = this->agent->get_coordinator();
	double max_distance = 0.0;
	int max_index = -1;
	double max_arrival_time = 0.0;
	double max_completion_time = 0.0;
	double max_arr_reward = -double(INFINITY);
	double max_comp_reward = -double(INFINITY);
	double max_comp_impact = -double(INFINITY);
	bool need_path = false;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			double e_dist = double(INFINITY);
			// get euclidean dist first
			if (world->dist_between_nodes(this->agent->get_edge().x, i, e_dist)) {
				double e_time = e_dist / this->agent->get_travel_vel();
				double w_time = this->world->get_nodes()[i]->get_time_to_complete(this->agent, this->world);
				double e_reward = this->world->get_nodes()[i]->get_reward_at_time(world->get_c_time() + e_time + w_time);
				// is my euclidean travel time reward better?
				if (e_reward > max_comp_impact) {
					// I am euclidean reward better, check a star
					std::vector<int> path;
					double a_dist = double(INFINITY);
					if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
						// am I a star better?
						double arr_time = this->world->get_c_time() + a_dist / this->agent->get_travel_vel();
						double arr_reward = this->world->get_nodes()[i]->get_reward_at_time(arr_time);
						double comp_time = arr_time + w_time;
						double comp_reward = this->world->get_nodes()[i]->get_reward_at_time(comp_time);
						// is it still the best with A*?
						if (comp_reward > max_comp_impact) {
							// is it taken by someone else?
							double prob_taken = 0.0;
							if (my_coord->get_advertised_task_claim_probability(i, comp_time, prob_taken, this->world)) {
								if (prob_taken == 0.0) {
									double impact = my_coord->get_reward_impact(i, this->agent->get_index(), comp_time, this->world);
									if (impact > max_comp_impact) {}
									// if not taken, then accept as possible goal
									max_comp_impact = impact;
									max_comp_reward = comp_reward;
									max_arr_reward = arr_reward;
									max_distance = a_dist;
									max_index = i;
									max_arrival_time = arr_time;
									max_completion_time = comp_time;
								}
								else {
									int a = i + 1;
								}
							}
						}
					}
				}
			}
		}
	}
	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(max_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(max_arrival_time);

		args.push_back("completion_time");
		vals.push_back(max_completion_time);

		args.push_back("arrival_reward");
		vals.push_back(max_arr_reward);

		args.push_back("completion_reward");
		vals.push_back(max_comp_reward);

		this->set_goal(max_index, args, vals);
	}

}

void Agent_Planning::select_task_by_impact_completion_value() {
	// select task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete - (travel_time + work_time)
	// select task by impact reward at time of completion, impact_reward = reward(t_complete) - reward(t^{next closest agent}_complete
	Agent_Coordinator* my_coord = this->agent->get_coordinator();
	double max_distance = 0.0;
	int max_index = -1;
	double max_arrival_time = 0.0;
	double max_completion_time = 0.0;
	double max_arr_reward = -double(INFINITY);
	double max_comp_reward = -double(INFINITY);
	double max_comp_impact = -double(INFINITY);
	bool need_path = false;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			double e_dist = double(INFINITY);
			// get euclidean dist first
			if (world->dist_between_nodes(this->agent->get_edge().x, i, e_dist)) {
				double e_time = e_dist / this->agent->get_travel_vel();
				double w_time = this->world->get_nodes()[i]->get_time_to_complete(this->agent, this->world);
				double e_reward = this->world->get_nodes()[i]->get_reward_at_time(world->get_c_time() + e_time + w_time);
				double e_value = e_reward - (e_time + w_time);
				// is my euclidean travel time reward better?
				if (e_value > max_comp_impact) {
					// I am euclidean reward better, check a star
					std::vector<int> path;
					double a_dist = double(INFINITY);
					if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
						// am I a star better?
						double arr_time = this->world->get_c_time() + a_dist / this->agent->get_travel_vel();
						double arr_reward = this->world->get_nodes()[i]->get_reward_at_time(arr_time);
						double comp_time = arr_time + w_time;
						double comp_reward = this->world->get_nodes()[i]->get_reward_at_time(comp_time);
						double comp_value = comp_reward - (a_dist / this->agent->get_travel_vel() + w_time);
						// is it still the best with A*?
						if (comp_value > max_comp_impact) {
							// is it taken by someone else?
							double prob_taken = 0.0;
							if (my_coord->get_advertised_task_claim_probability(i, comp_time, prob_taken, this->world)) {
								if (prob_taken == 0.0) {
									double impact = my_coord->get_reward_impact(i, this->agent->get_index(), comp_time, this->world);
									double impact_value = impact - (a_dist / this->agent->get_travel_vel() + w_time);
									if (impact_value > max_comp_impact) {}
									// if not taken, then accept as possible goal

									max_comp_impact = impact_value;
									max_comp_reward = comp_reward;
									max_arr_reward = arr_reward;
									max_distance = a_dist;
									max_index = i;
									max_arrival_time = arr_time;
									max_completion_time = comp_time;
								}
								else {
									int a = i + 1;
								}
							}
						}
					}
				}
			}
		}
	}
	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(max_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(max_arrival_time);

		args.push_back("completion_time");
		vals.push_back(max_completion_time);

		args.push_back("arrival_reward");
		vals.push_back(max_arr_reward);

		args.push_back("completion_reward");
		vals.push_back(max_comp_reward);

		this->set_goal(max_index, args, vals);
	}
}


void Agent_Planning::MCTS_task_by_completion_reward() {
	this->MCTS_task_selection();
	//std::cout << "mcts_by_comp_reward::planning_time: " << double(clock()) / double(CLOCKS_PER_SEC) - s_time << std::endl;
}

void Agent_Planning::D_MCTS_task_by_completion_reward() {
	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: in 'D_MCTS_task_selection' on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	double reward_in = 0.0;
	double s_time = double(clock()) / double(CLOCKS_PER_SEC);
	std::vector<bool> task_list; // list of all tasks status, true if active, false if complete
	std::vector<int> task_set; // list, by index, of all active tasks
	this->world->get_task_status_list(task_list, task_set);
	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: got task_list (%i) / task_set (%i)", int(task_list.size()), int(task_set.size()));
	if (!this->dmcts) {
		ROS_WARN("Agent_Planning::D_MCTS_task_by_completion_reward: initializing dmcts");
		this->dmcts = new D_MCTS(this->world, this->world->get_nodes()[this->get_agent()->get_loc()], this->get_agent(), NULL, 0, this->world->get_c_time());
	}
	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: finished initializing D_MCTS");

	//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: going into search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	// task of root is marked complete
	//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: set task index (%i) to false", this->dmcts->get_task_index());
	task_list[this->dmcts->get_task_index()] = false;
	//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: set task index (%i) to false !!!!!!!!", this->dmcts->get_task_index());
	//while( double(clock()) / double(CLOCKS_PER_SEC) - s_time <= this->reoccuring_search_time){
		this->planning_iter++;
		int depth_in = 0;
		//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: really going into search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
		this->dmcts->search_from_root(task_list, task_set);
	//	if( planning_iter % 1000 == 0){
			this->dmcts->update_probable_actions();
	//	}
		//ROS_INFO("Agent_Planning::D_MCTS_task_by_completion_reward: out of search on edge %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
	//}

	//ROS_INFO("Agent_Planning::D_MCTS_task_selection: finished searching tree for 1 inter");
	int planning_iters = this->planning_iter - this->last_planning_iter_end;
	//std::cout << "planning_iters: " << planning_iters << std::endl;
	this->last_planning_iter_end = this->planning_iter;

	this->agent->get_coordinator()->reset_prob_actions(); // clear out probable actions before adding the new ones
	this->dmcts->sample_tree_and_advertise_task_probabilities(this->agent->get_coordinator());
	//printf("sampling time: %0.2f \n", double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	std::vector<int> best_path;
	std::vector<double> times;
	std::vector<double> rewards;
	this->dmcts->get_best_path(best_path, times, rewards);
	
	//std::vector<double> probs(int(best_path.size()), 1.0);
	//this->agent->get_coordinator()->upload_new_plan(best_path, times, probs);
	//ROS_WARN("Agent_Planning: planning_iter %i and this iters: %i", this->planning_iter, planning_iters);

	std::cout << "Agent_Planning[" << this->agent->get_index() << "]: best_path: ";
	for(size_t i=0; i<best_path.size(); i++){
		std::cout << std::fixed << std::setprecision(2) << " ( Path[" << i << "]: " << best_path[i] << " @ " << times[i] << " for " << rewards[i] <<"), ";// << " with probs: " << probs[i] << "), ";
	}
	std::cout << std::endl;
	
	/*
	std::ofstream outfile;
	outfile.open("planning_time.txt", std::ios::app);
	char buffer[50];
	int n = sprintf_s(buffer, "%i, %0.6f\n", planning_iters, double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	outfile << buffer;
	outfile.close();
	*/

	ROS_INFO("Agent[%i]'s coord tree", this->agent->get_index());
	this->agent->get_coordinator()->print_prob_actions();
	for(int i=0; i<this->world->get_n_agents(); i++){
		if(i != this->agent->get_index()){
			ROS_INFO("Agent[%i]'s understanding of agent[%i]s coord tree", this->agent->get_index(), i);
			this->world->get_agents()[i]->get_coordinator()->print_prob_actions();
		}
	}
	
	
	//? - comeback to this after below: why does planning iter for agent 0 only do a few iters but for agent 1 it does 100s?

	if(this->world->get_c_time() > this->initial_search_time){
		if (this->agent->get_at_node()) {
			// I am at either edge.x / edge.y
			int max_kid_index = -1;
			std::vector<std::string> args;
			std::vector<double> vals;
			//ROS_ERROR("Agent_Planning::D_MCTS_task_selection: I am at node: %i/%i with goal: %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index());
			if (this->dmcts->exploit_tree(max_kid_index, args, vals)) {
				//std::cerr << "max_kid_index: " << max_kid_index << std::endl;
				int goal_task_index = this->dmcts->get_kids()[max_kid_index]->get_task_index();
				//ROS_INFO("Agent_Planning::D_MCTS_task_selection: exploit_tree succesful with max_kid_index %i and task index %i", max_kid_index, goal_task_index);
				// only make a new goal if the current goai is NOT active
				//ROS_INFO("Agent_Planning::D_MCTS_task_selection: checking if current goal (%i) is active, it is %i", this->agent->get_goal()->get_index(), this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active());
				if(this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active() == 0){
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: goal is NOT active");
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: I am at node: %i/%i with A COMPLETED goal: %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index());
					this->set_goal(goal_task_index);
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: set new max_kid_index: %i", goal_task_index);
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: resetting dmcts root");
					this->dmcts->prune_branches(max_kid_index);
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: pruned branches");
					D_MCTS* old = this->dmcts;
					this->dmcts = this->dmcts->get_kids()[max_kid_index];
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: got_kids");
					this->dmcts->set_as_root();
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: reset dmcts root");
				}

				if (!world->are_nbrs(this->agent->get_loc(), goal_task_index) ) {
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: not nbrs");
					// I am not nbrs with the next node (my goal), replace root index with current node but don't advance/prune tree
					this->dmcts->set_task_index(this->agent->get_edge().y);
					//ROS_INFO("Agent_Planning::D_MCTS_task_selection: set task index to %i", this->agent->get_edge().y);
				}	
			}
		}
	}
	//ROS_WARN("Agent_Planning::D_MCTS_task_selection: edge out %i -> %i", this->agent->get_edge().x, this->agent->get_edge().y);
}


void Agent_Planning::MCTS_task_by_completion_value() {
	this->MCTS_task_selection();
	//std::cout << "mcts_by_comp_value::planning_time: " << double(clock()) / double(CLOCKS_PER_SEC) - s_time << std::endl;
}

void Agent_Planning::MCTS_task_by_completion_reward_impact() {
	this->MCTS_task_selection();
}

void Agent_Planning::MCTS_task_by_completion_value_impact() {
	//ROS_INFO("Agent_Planning::MCTS_task_by_completion_value_impact: going into 'MCTS_task_selection'");
	this->MCTS_task_selection();
	//ROS_INFO("Agent_Planning::MCTS_task_by_completion_value_impact: out of 'MCTS_task_selection'");
	//std::cout << "mcts_by_comp_value::planning_time: " << double(clock()) / double(CLOCKS_PER_SEC) - s_time << std::endl;
}

void Agent_Planning::MCTS_task_selection(){
	//ROS_INFO("Agent_Planning::MCTS_task_selection: in 'MCTS_task_selection'");
	double reward_in = 0.0;
	double s_time = double(clock()) / double(CLOCKS_PER_SEC);
	std::vector<bool> task_list;
	std::vector<int> task_set;
	this->world->get_task_status_list(task_list, task_set);
	//ROS_INFO("Agent_Planning::MCTS_task_selection: got task_list (%i) / task_set (%i)", int(task_list.size()), int(task_set.size()));
	if (!this->mcts) {
		this->mcts = new MCTS(this->world, this->world->get_nodes()[this->get_agent()->get_loc()], this->get_agent(), NULL, 0, this->world->get_c_time());
	}
	//ROS_INFO("Agent_Planning::MCTS_task_selection: finished initializing MCTS");

	// TODO implement moving average on probability of bid

	// task of root is marked complete
	task_list[this->mcts->get_task_index()] = false;
	while( double(clock()) / double(CLOCKS_PER_SEC) - s_time <= this->reoccuring_search_time){
		this->planning_iter++;
		int depth_in = 0;
		this->mcts->search_from_root(task_list, task_set, last_planning_iter_end, planning_iter);
	}
	//ROS_INFO("Agent_Planning::MCTS_task_selection: finished searching tree for 1 inter");
	int planning_iters = this->planning_iter - this->last_planning_iter_end;
	std::cout << "planning_iters: " << planning_iters << std::endl;
	this->last_planning_iter_end = this->planning_iter;

	this->agent->get_coordinator()->reset_prob_actions(); // clear out probable actions before adding the new ones
	this->mcts->sample_tree_and_advertise_task_probabilities(this->agent->get_coordinator());
	//printf("sampling time: %0.2f \n", double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	std::vector<int> best_path;
	std::vector<double> times;
	std::vector<double> rewards;
	this->mcts->get_best_path(best_path, times, rewards);
	
	//std::vector<double> probs(int(best_path.size()), 1.0);
	//this->agent->get_coordinator()->upload_new_plan(best_path, times, probs);
	//ROS_WARN("Agent_Planning: planning_iter %i and this iters: %i", this->planning_iter, planning_iters);

	//std::cout << "Agent_Planning[" << this->agent->get_index() << "]: best_path: ";
	//for(size_t i=0; i<best_path.size(); i++){
	//	std::cout << " (" << i << ": " << best_path[i] << " @ " << times[i] << " for " << rewards[i] <<"), ";// << " with probs: " << probs[i] << "), ";
	//}
	//std::cout << std::endl;
	
	/*
	std::ofstream outfile;
	outfile.open("planning_time.txt", std::ios::app);
	char buffer[50];
	int n = sprintf_s(buffer, "%i, %0.6f\n", planning_iters, double(clock()) / double(CLOCKS_PER_SEC) - s_time);
	outfile << buffer;
	outfile.close();
	*/

	/*
	ROS_INFO("Agent[%i]'s coord tree", this->agent->get_index());
	this->agent->get_coordinator()->print_prob_actions();
	for(int i=0; i<this->world->get_n_agents(); i++){
		if(i != this->agent->get_index()){
			ROS_INFO("Agent[%i]'s understanding of agent[%i]s coord tree", this->agent->get_index(), i);
			this->world->get_agents()[i]->get_coordinator()->print_prob_actions();
		}
	}
	*/
	
	//? - comeback to this after below: why does planning iter for agent 0 only do a few iters but for agent 1 it does 100s?

	if(this->world->get_c_time() > this->initial_search_time){
		if (this->agent->get_at_node()) {
			// I am at either edge.x / edge.y
			int max_index;
			std::vector<std::string> args;
			std::vector<double> vals;
			ROS_ERROR("Agent_Planning::MCTS_task_selection: I am at node: %i/%i with goal: %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index());
			//std::cout << "Agent_Planning[" << this->agent->get_index() << "]: best_path: ";
			//for(size_t i=0; i<best_path.size(); i++){
			//	std::cout << " (" << i << ": " << best_path[i] << " @ " << times[i] << " for " << rewards[i] << " with probs: " << probs[i] << "), ";
			//}
			//std::cout <<std::endl;
			if (this->mcts->exploit_tree(max_index, args, vals)) {
				ROS_INFO("Agent_Planning::MCTS_task_selection: exploit_tree succesful with max_index: %i", max_index);
				// only make a new goal if the current goai is NOT active
				ROS_INFO("Agent_Planning::MCTS_task_selection: checking if %i is active, it is %i", this->agent->get_goal()->get_index(), this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active());
				if(this->world->get_nodes()[this->agent->get_goal()->get_index()]->is_active() == 0){
					ROS_WARN("Agent_Planning::MCTS_task_selection: goal is NOT active");
					ROS_INFO("Agent_Planning::MCTS_task_selection: I am at node: %i/%i with A COMPLETED goal: %i", this->agent->get_edge().x,this->agent->get_edge().y, this->agent->get_goal()->get_index());
					this->set_goal(max_index);
					ROS_INFO("Agent_Planning::MCTS_task_selection: am nbrs with next node, resetting mcts root");
					this->mcts->prune_branches();
					MCTS* old = this->mcts;
					this->mcts = this->mcts->get_golden_child();
					this->mcts->set_as_root();
				}

				if (world->are_nbrs(this->agent->get_loc(), max_index) ) {
					// I am nbrs with the next node (my goal) in the tree. Replace root with child and prune

				}
				else {
					// I am not nbrs with the next node (my goal), replace root index with current node but don't advance/prune tree
					this->mcts->set_task_index(this->agent->get_edge().y);
					/*
					int ind = int(this->agent->get_goal()->get_path().size()) - 2; //
					ROS_INFO("Agent_Planning::MCTS_task_selection: I am NOT nbrs with next node");
					if (ind >= 0) {
						// update the task index of the first node
						ROS_INFO("Agent_Planning::MCTS_task_selection: updating task index");
						this->mcts->set_task_index(this->agent->get_goal()->get_path()[ind]);
					}
					else if (this->agent->get_goal()->get_path().size() == 1) {
						ROS_INFO("Agent_Planning::MCTS_task_selection: weird at node behavior");
						this->mcts->set_task_index(this->agent->get_goal()->get_index());
					}
					*/
				}	
			}
		}
	}
}

void Agent_Planning::set_goal(const int &goal_index) {
	// create a new goal from scratch, reset everything!
	//ROS_INFO("Agent_Planning::set_goal: goal_index: %i from cLoc: %i", goal_index, this->agent->get_edge().x);
	this->agent->get_goal()->set_index(goal_index);
	std::vector<int> path;
	double length = 0.0;
	bool need_path = true;
	if (world->a_star(this->agent->get_edge().x, this->agent->get_goal()->get_index(), this->agent->get_pay_obstacle_cost(), need_path, path, length)) {
		this->agent->get_goal()->set_distance(length);
		this->agent->get_goal()->set_path(path);
		//std::cout << "Agent_Planning::set_goal::Path to goal: ";
		//for(size_t i=0; i<path.size(); i++){
		//	std::cout << path[i] << ", ";
		//}
		//std::cout << std::endl;
	}
	else {
		this->agent->get_goal()->set_distance(double(INFINITY));
		ROS_WARN("Agent_Planning:set_goal: A* could not find path to node");
	}
	this->agent->get_goal()->set_current_time(world->get_c_time());
	double travel_time = this->agent->get_goal()->get_distance() / (this->agent->get_travel_vel()*world->get_dt());
	double arrival_time = this->agent->get_goal()->get_current_time() + travel_time;
	this->agent->get_goal()->set_arrival_time(arrival_time);
	double work_time = this->world->get_nodes()[goal_index]->get_time_to_complete(this->agent, this->world);
	this->agent->get_goal()->set_completion_time(world->get_c_time() + this->agent->get_goal()->get_arrival_time() + work_time);
	this->agent->get_goal()->set_current_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_current_time()));
	this->agent->get_goal()->set_arrival_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_arrival_time()));
	this->agent->get_goal()->set_completion_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_completion_time()));
}


void Agent_Planning::set_goal(const int &goal_index, std::vector<int> &path) {
	// create a new goal from scratch, reset everything!
	ROS_INFO("Agent_Planning::set_goal: goal_index: %i from cLoc: %i", goal_index, this->agent->get_edge().x);
	this->agent->get_goal()->set_index(goal_index);
	path.clear();
	double length = 0.0;
	bool need_path = true;
	if (world->a_star(this->agent->get_edge().x, this->agent->get_goal()->get_index(), this->agent->get_pay_obstacle_cost(), need_path, path, length)) {
		this->agent->get_goal()->set_distance(length);
		this->agent->get_goal()->set_path(path);
		std::cout << "Agent_Planning::set_goal::Path to goal: ";
		for(size_t i=0; i<path.size(); i++){
			std::cout << path[i] << ", ";
		}
		std::cout << std::endl;
	}
	else {
		this->agent->get_goal()->set_distance(double(INFINITY));
		ROS_WARN("Agent_Planning:set_goal: A* could not find path to node");
	}
	this->agent->get_goal()->set_current_time(world->get_c_time());
	double travel_time = this->agent->get_goal()->get_distance() / (this->agent->get_travel_vel()*world->get_dt());
	double arrival_time = this->agent->get_goal()->get_current_time() + travel_time;
	this->agent->get_goal()->set_arrival_time(arrival_time);
	double work_time = this->world->get_nodes()[goal_index]->get_time_to_complete(this->agent, this->world);
	this->agent->get_goal()->set_completion_time(world->get_c_time() + this->agent->get_goal()->get_arrival_time() + work_time);
	this->agent->get_goal()->set_current_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_current_time()));
	this->agent->get_goal()->set_arrival_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_arrival_time()));
	this->agent->get_goal()->set_completion_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_completion_time()));
}

void Agent_Planning::set_goal(const int &goal_index, const std::vector<std::string> &args, const std::vector<double> &vals) {
	
	bool need_distance = true;
	bool need_current_reward = true;
	bool need_arrival_reward = true;
	bool need_completion_reward = true;
	
	bool need_current_time = true;
	bool need_arrival_time = true;
	bool need_completion_time = true;

	bool need_completion_value = true;
	

	for (size_t a = 0; a < args.size(); a++) { // check through all args
		if (args[a].compare("distance") == 0) {
			this->agent->get_goal()->set_distance(vals[a]);
			need_distance = false;
		}
		else if (args[a].compare("current_reward") == 0) {
			this->agent->get_goal()->set_current_reward(vals[a]);
			need_current_reward = false;
		}
		else if (args[a].compare("arrival_reward") == 0) {
			this->agent->get_goal()->set_arrival_reward(vals[a]);
			need_arrival_reward = false;
		}
		else if (args[a].compare("completion_reward") == 0) {
			this->agent->get_goal()->set_completion_reward(vals[a]);
			need_completion_reward = false;
		}
		else if (args[a].compare("current_time") == 0) {
			this->agent->get_goal()->set_current_time(vals[a]);
			need_current_time = false;
		}
		else if (args[a].compare("arrival_time") == 0) {
			this->agent->get_goal()->set_arrival_time(vals[a]);
			need_arrival_time = false;
		}
		else if (args[a].compare("completion_time") == 0) {
			this->agent->get_goal()->set_completion_time(vals[a]);
			need_completion_time = false;
		}
		else if (args[a].compare("completion_value") == 0) {
			this->agent->get_goal()->set_completion_value(vals[a]);
			need_completion_value = false;
		}
		else {
			std::cerr << "Agent_Planning::set_goal: bad arg: " << args[a] << std::endl;
		}
	}
	
	this->agent->get_goal()->set_index(goal_index);
	
	if (need_distance) {
		std::vector<int> path;
		bool need_path = false;
		double length = 0.0;
		if (world->a_star(this->agent->get_edge().x, this->agent->get_goal()->get_index(), this->agent->get_pay_obstacle_cost(), need_path, path, length)) {
			this->agent->get_goal()->set_distance(length);
		}
		else {
			this->agent->get_goal()->set_distance(double(INFINITY));
			ROS_ERROR("Agent_Planning:set_goal: A* could not find path to node");
		}
	}

	if (need_current_time) {
		this->agent->get_goal()->set_current_time(world->get_c_time());
	}
	if (need_current_time) {
		this->agent->get_goal()->set_current_time(world->get_c_time());
	}
	if (need_arrival_time) {
		double travel_time = this->agent->get_goal()->get_distance() / (this->agent->get_travel_vel()*world->get_dt());
		this->agent->get_goal()->set_arrival_time(this->agent->get_goal()->get_current_time() + travel_time);
	}
	if (need_completion_time) {
		double work_time = this->world->get_nodes()[goal_index]->get_time_to_complete(this->agent, this->world);
		this->agent->get_goal()->set_completion_time(world->get_c_time() + this->agent->get_goal()->get_arrival_time()+ work_time);
	}
	if (need_current_reward) {
		this->agent->get_goal()->set_current_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_current_time()));
	}
	if (need_arrival_reward) {
		this->agent->get_goal()->set_arrival_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_arrival_time()));
	}
	if (need_completion_reward) {
		this->agent->get_goal()->set_completion_reward(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_completion_time()));
	}
	if (need_completion_value) {
		this->agent->get_goal()->set_completion_value(world->get_nodes()[goal_index]->get_reward_at_time(this->agent->get_goal()->get_completion_time()) - (this->agent->get_goal()->get_completion_time() - this->world->get_c_time()));
	}
}

void Agent_Planning::select_random_nbr() {
	if (!this->agent->get_at_node()) {
		return;
	}
	int goal_index;
	int n_nbrs = this->world->get_nodes()[this->agent->get_edge().x]->get_n_nbrs();
	int c_nbr = rand() % n_nbrs;
	world->get_nodes()[this->agent->get_edge().x]->get_nbr_i(c_nbr, goal_index);

	this->set_goal(goal_index);
}

void Agent_Planning::select_random_node() {
	if (!this->agent->get_at_node()) {
		return;
	}
	int n_nodes = this->world->get_n_nodes();
	int goal_index = rand() % n_nodes;

	this->set_goal(goal_index);
}

void Agent_Planning::select_random_task() {
	if (!this->agent->get_at_node()) {
		return;
	}
	int c_goal = this->agent->get_goal()->get_index();
	if (world->get_nodes()[c_goal]->is_active()) {
		return;
	}

	std::vector<int> active_tasks;
	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			active_tasks.push_back(i);
		}
	}

	if (active_tasks.size() > 0) {
		int goal_index = active_tasks[rand() % int(active_tasks.size())];
		this->set_goal(goal_index);
	}
}

void Agent_Planning::select_greedy_task_by_current_reward() {
	if (!this->agent->get_at_node()) {
		return;
	}
	double c_time = std::clock() / double(CLOCKS_PER_SEC);

	double max_reward = -double(INFINITY);
	int max_index = -1;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			double t_reward = this->world->get_nodes()[i]->get_reward_at_time(c_time);
			double prob_taken = 0.0;
			if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, this->world->get_c_time(), prob_taken, this->world)) {
				if (t_reward*(1 - prob_taken) > max_reward) {
					max_reward = t_reward*(1 - prob_taken);
					max_index = i;
				}
			}
		}
	}
	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		
		args.push_back("current_reward");
		vals.push_back(max_reward);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		this->set_goal(max_index, args, vals);
	}
}


void Agent_Planning::select_greedy_task_by_arrival_reward() {
	if (!this->agent->get_at_node()) {
		return;
	}
	double max_distance = 0.0;
	int max_index = -1;
	double max_arrival_time = 0.0;
	double max_arr_reward = -double(INFINITY);
	bool need_path = false;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			std::vector<int> path;
			double a_dist = double(INFINITY);
			if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
				// am I a star better?
				double arr_time = this->world->get_c_time() + a_dist / this->agent->get_travel_vel();
				double arr_reward = this->world->get_nodes()[i]->get_reward_at_time(arr_time);
				if (arr_reward > max_arr_reward) {
					// is it taken by someone else?
					double prob_taken = 0.0;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, arr_time, prob_taken, this->world)) {
						// if not taken, then accept as possible goal
						arr_reward *= (1.0 - prob_taken);
						if (arr_reward > max_arr_reward) {
							max_arr_reward = arr_reward;
							max_distance = a_dist;
							max_index = i;
							max_arrival_time = arr_time;
						}
					}
				}
			}
		}
	}
	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(max_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(max_arrival_time);

		args.push_back("arrival_reward");
		vals.push_back(max_arr_reward);

		this->set_goal(max_index, args, vals);
	}
}

void Agent_Planning::select_greedy_task_by_completion_reward() {

	//ROS_ERROR("Agent_Planning[%i]::select_greedy_task_by_completion_reward: in",this->agent->get_index());
	if (!this->agent->get_at_node()) {
		return;
	}
	//ROS_ERROR("Agent_Planning[%i]::select_greedy_task_by_completion_reward: in 2",this->agent->get_index());

	double max_distance = 0.0;
	int max_index = -1;
	double max_arrival_time = 0.0;
	double max_completion_time = 0.0;
	double max_arr_reward = -double(INFINITY);
	double max_comp_reward = -double(INFINITY);
	bool need_path = false;

	//std::cerr << "world n_nodes: " << this->world->get_n_nodes() << std::endl;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		//std::cerr << "i: " << i;
		if (world->get_nodes()[i]->is_active()) {
			//std::cerr << "node " << i << " is active" << std::endl;
			std::vector<int> path;
			double a_dist = double(INFINITY);
			if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
				//std::cerr << "a_dist: " << a_dist << std::endl;
				// am I a star better?
				double arr_time = this->world->get_c_time() + a_dist / this->agent->get_travel_vel();
				double arr_reward = this->world->get_nodes()[i]->get_reward_at_time(arr_time);
				double w_time = this->world->get_nodes()[i]->get_time_to_complete(this->agent, this->world);
				double comp_time = arr_time + w_time;
				double comp_reward = this->world->get_nodes()[i]->get_reward_at_time(comp_time);
				//std::cerr << "comp_reward: " << comp_reward << std::endl;
				if (comp_reward > max_comp_reward) {
					// is it taken by someone else?
					double prob_taken = 0.0;
					//std::cerr << "comp_reward > max_comp_reward" << std::endl;
					//std::cerr << "comp_time: " << comp_time << std::endl;
					//std::cerr << "prob_taken: " << prob_taken << std::endl;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, comp_time, prob_taken, this->world)) {
						// if not taken, then accept as possible goal
						//std::cerr << "prob_taken: " << prob_taken << std::endl;
						comp_reward *= (1.0 - prob_taken);
						if (comp_reward > max_comp_reward) {
							max_comp_reward = comp_reward;
							max_arr_reward = arr_reward;
							max_distance = a_dist;
							max_index = i;
							max_arrival_time = arr_time;
							max_completion_time = comp_time;
						}
					}
				}
			}
		}
	}


	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(max_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(max_arrival_time);

		args.push_back("completion_time");
		vals.push_back(max_completion_time);

		args.push_back("arrival_reward");
		vals.push_back(max_arr_reward);

		args.push_back("completion_reward");
		vals.push_back(max_comp_reward);

		this->set_goal(max_index, args, vals);
	}
}

void Agent_Planning::select_task_by_current_value() {
	if (!this->agent->get_at_node()) {
		return;
	}
	// probably don't do these
}

void Agent_Planning::select_task_by_arrival_value() {
	if (!this->agent->get_at_node()) {
		return;
	}
	// probably don't do these
}

void Agent_Planning::select_task_by_completion_value() {
	if (!this->agent->get_at_node()) {
		return;
	}
	// what is the value I will recieve for completing this task -> value = c_0*reward(t_complete) - c_1*(travel_time + work_time)
	double max_distance = 0.0;
	int max_index = -1;
	double max_arrival_time = 0.0;
	double max_completion_time = 0.0;
	double max_arr_reward = 0.0;
	double max_comp_reward = 0.0;
	double max_comp_value = -double(INFINITY);
	bool need_path = false;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			std::vector<int> path;
			double a_dist = double(INFINITY);
			if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
				// am I a star better?
				double a_time = a_dist / this->agent->get_travel_vel();
				double arr_time = this->world->get_c_time() + a_time;
				double arr_reward = this->world->get_nodes()[i]->get_reward_at_time(arr_time);
				double w_time = this->world->get_nodes()[i]->get_time_to_complete(this->agent, this->world);
				double comp_time = arr_time + w_time;
				double comp_reward = this->world->get_nodes()[i]->get_reward_at_time(comp_time);
				double comp_value = comp_reward - (a_time + w_time);
				if (comp_value > max_comp_value) {
					// is it taken by someone else?
					double prob_taken = 0.0;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, comp_time, prob_taken, this->world)) {
						// if not taken, then accept as possible goal
						comp_value = comp_reward*(1.0-prob_taken) - (a_time + w_time);
						if (comp_value > max_comp_value) {
							max_comp_reward = comp_reward;
							max_arr_reward = arr_reward;
							max_distance = a_dist;
							max_index = i;
							max_arrival_time = arr_time;
							max_completion_time = comp_time;
							max_comp_value = comp_value;
						}
					}
				}
			}
		}
	}

	if (max_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(max_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(max_arrival_time);

		args.push_back("completion_time");
		vals.push_back(max_completion_time);

		args.push_back("arrival_reward");
		vals.push_back(max_arr_reward);

		args.push_back("completion_reward");
		vals.push_back(max_comp_reward);

		args.push_back("completion_value");
		vals.push_back(max_comp_value);

		this->set_goal(max_index, args, vals);
	}
}

void Agent_Planning::select_task_by_reward_impact() {
	if (!this->agent->get_at_node()) {
		return;
	}
	// choose the task with the highest reward impact. Where impact is the difference in
	// the team reward if I get it now instead of not getting it and waiting on someone else, the
	// difference between now and the next claim or potentially all following claims
}

void Agent_Planning::select_task_by_value_impact() {
	if (!this->agent->get_at_node()) {
		return;
	}
	// choose the task with the highest value impact. Where impact is the difference in
	// the team reward if I get it now instead of not getting it and waiting on someone else, the
	// difference between now and the next claim or potentially all following claims

}
void Agent_Planning::select_greedy_task_by_completion_time() {
	if (!this->agent->get_at_node()) {
		return;
	}
	double min_distance = double(INFINITY);
	int min_index = -1;
	double min_arrival_time = double(INFINITY);
	double min_completion_time = double(INFINITY);
	bool need_path = false;


	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			std::vector<int> path;
			double a_dist = double(INFINITY);
			if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
				// am I a star closer?
				if (a_dist < min_distance) {

					double a_time = this->world->get_c_time() + a_dist / (this->agent->get_travel_vel() * this->world->get_dt());
					double w_time = this->world->get_nodes()[i]->get_time_to_complete(this->agent, this->world);
					double c_time = a_time + w_time;
					// is it taken by someone else?
					double prob_taken = 0.0;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, c_time, prob_taken, this->world)) {
						// if not taken, then accept as possible goal
						if (prob_taken == 0.0) {
							min_distance = a_dist;
							min_index = i;
							min_arrival_time = a_time;
							min_completion_time = c_time;
						}
					}
				}
			}
		}
	}
	if (min_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		vals.push_back(min_distance);

		args.push_back("current_time");
		vals.push_back(world->get_c_time());

		args.push_back("arrival_time");
		vals.push_back(min_arrival_time);

		args.push_back("completion_time");		
		vals.push_back(min_completion_time);

		this->set_goal(min_index, args, vals);
	}
}

void Agent_Planning::select_greedy_task_by_arrival_time() {
	if (!this->agent->get_at_node()) {
		return;
	}
	double min_distance = double(INFINITY);
	int min_index = -1;
	double min_arrival_time = double(INFINITY);
	bool need_path = false;

	for (int i = 0; i < this->world->get_n_nodes(); i++) {
		if (world->get_nodes()[i]->is_active()) {
			std::vector<int> path;
			double a_dist = double(INFINITY);
			if (world->a_star(this->agent->get_edge().x, i, this->agent->get_pay_obstacle_cost(), need_path, path, a_dist)) {
				// am I a star closer?
				if (a_dist < min_distance) {
					double a_time = this->world->get_c_time() + a_dist / (this->agent->get_travel_vel() * this->world->get_dt());
					// is it taken by someone else?
					double prob_taken = 0.0;
					if (this->agent->get_coordinator()->get_advertised_task_claim_probability(i, a_time, prob_taken, this->world)) {
						// if not taken, then accept as possible goal
						if (prob_taken == 0.0) {
							min_distance = a_dist;
							min_index = i;
							min_arrival_time = a_time;
						}
					}
				}
			}
		}
	}

	if (min_index > -1) {
		std::vector<std::string> args;
		std::vector<double> vals;
		args.push_back("distance");
		args.push_back("current_time");
		args.push_back("arrival_time");
		vals.push_back(min_distance);
		vals.push_back(min_arrival_time);
		vals.push_back(world->get_c_time());
		this->set_goal(min_index, args, vals);
	}
}

