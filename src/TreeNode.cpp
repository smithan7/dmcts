#include "TreeNode.h"

#include <iostream>

#include "World.h"
#include "Agent.h"
#include "Map_Node.h"
#include "Agent_Coordinator.h"


TreeNode::TreeNode(World* world, Map_Node* task_in, Agent* agent_in, int depth_in) {
	this->my_depth = depth_in;
	this->agent = agent_in;
	this->task = task_in;

	if (this->my_depth == 0) {
		this->my_probability = 1.0;
	}
	this->wait_time = 5.0f;
	this->value = 0.0;
	this->max_child_index = -1;
	this->max_child_value = -1.0;
	this->sum_child_value = -1.0;
	this->nPulls = 0;
	this->my_probability = 0.0;
}

TreeNode::~TreeNode() {
	// clear children
	for (size_t i = 0; i < this->children.size(); i++) {
		delete this->children[i];
	}
	this->children.clear();
}

void TreeNode::make_children(const std::vector<bool> &task_status) {
	if (world) { // has tasks and not NULL

		// add a child who stands still
		TreeNode* waiting_tree = new TreeNode(world, this->task, this->agent, this->my_depth + 1);
		children.push_back(waiting_tree);

		// add a child for each active task
		std::vector<Map_Node*> tasks = world->get_nodes();
		for (size_t i = 0; i < tasks.size(); i++) {
			// if task i needs to be completed
			if (task_status[i]) {
				TreeNode* tree = new TreeNode(world, tasks[i], this->agent, this->my_depth + 1);
				children.push_back(tree);
			}
		}
	}
	else {
		std::cerr << "TreeNode::make_children: ra not initialized" << std::endl;
	}
}

void TreeNode::make_children(const std::vector<int> &available_tasks) {
	if (world) { // has tasks and not NUL
		// add a child who stands still
		TreeNode* waiting_tree = new TreeNode(world, this->task, this->agent, this->my_depth + 1);
		children.push_back(waiting_tree);

		// add a child for each active task
		for (size_t i = 0; i < available_tasks.size(); i++) {
			TreeNode* tree = new TreeNode(world, world->get_nodes()[i], this->agent, this->my_depth + 1);
			children.push_back(tree);
		}
	}
	else {
		std::cerr << "TreeNode::make_children: ra not initialized" << std::endl;
	}
}

double TreeNode::evaluate_world_at_time(const double &time) {
	double penalty = 0;
	for (int t = 0; t < world->get_nodes().size(); t++) {
		if (world->get_nodes()[t]->is_active()) {
			penalty += world->get_nodes()[t]->get_reward_at_time(time);
		}
	}
	return penalty;
}

bool TreeNode::get_travel_and_task_completion_time(const int &location, const int &goal_task, const double &time_in, double &distance, double &completion_time) {
	// distance between me and the task
	double travel_time;
	if (world->get_travel_time(location, goal_task, agent->get_travel_step(), agent->get_pay_obstacle_cost(), travel_time)) {
		// when will I arrive
		double time_arrive_at_task = time_in + travel_time;
		// how long will i need to work on task
		double work_time;
		if (world->get_task_completion_time(this->agent->get_index(), goal_task, work_time)) {
			// return time I will leave task completed
			completion_time = work_time + time_arrive_at_task;
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

bool TreeNode::evaluate_at_time_except(const double &initial_time, const int &index_to_evaluate, double &task_completion_time, double &penalty) {

	double distance = 0.0;
	if (this->my_depth == 0) {
		// from my location, root node, to first task
		if (!this->get_travel_and_task_completion_time(this->agent->get_loc(), this->children[index_to_evaluate]->task->get_index(), initial_time, distance, task_completion_time)) {
			std::cerr << "TreeNode::evaluate_at_time_except::Could not get travel and task completion time" << std::endl;
		}
	}
	else {

		// first child is to stay put
		if (index_to_evaluate == 0) {
			// stay at current location
			task_completion_time = initial_time + 5.0f;
		}
		else {
			// from task to other task
			if (!this->get_travel_and_task_completion_time(this->task->get_index(), this->children[index_to_evaluate]->task->get_index(), initial_time, distance, task_completion_time)) {
				std::cerr << "TreeNode::evaluate_at_time_except::Could not get travel and task completion time" << std::endl;
				return false;
			}
		}
	}

	if (this->children.size() > 0) {
		for (int t = 0; t < this->children.size(); t++) {
			if (this->children[t]->task->get_index() != index_to_evaluate && world->get_nodes()[t]->is_active()) {
				//UE_LOG(LogTemp, Warning, TEXT("TreeNode::evaluate_at_time_except::Task %i is getting checked"), world->get_nodes()[t]->get_index());
				double raw_penalty = this->children[t]->task->get_reward_at_time(task_completion_time);
				double p_complete = get_coordinator_probability_for_task(t, task_completion_time);
				penalty += raw_penalty*(1 - p_complete);
			}
		}
		return true;
	}
	else {
		std::cerr << "TreeNode::evaluate_at_time_except::Children not initialized" << std::endl;
		return false;
	}
}

double TreeNode::child_value_at_time(const double &initial_time, const int &child_to_evaluate, double &task_completion_time) {

	// first child is to stay put
	if (child_to_evaluate == 0 || !this->children[child_to_evaluate]->task) {
		// stay at current location
		task_completion_time = initial_time + this->wait_time;
		return 0.0;
	}

	double distance = 0.0;
	if (this->my_depth == 0 || !this->task) {
		// from my location, root node, to first task
		if(!this->get_travel_and_task_completion_time(this->agent->get_loc(), this->children[child_to_evaluate]->task->get_index(), initial_time, distance, task_completion_time)){
			std::cerr << "TreeNode::child_value_at_time::Could not get travel and task completion time" << std::endl;
		}
	}
	else {
		// from task to other task
		if (!this->get_travel_and_task_completion_time(this->task->get_index(), this->children[child_to_evaluate]->task->get_index(), initial_time, distance, task_completion_time)) {
			std::cerr << "TreeNode::child_value_at_time::Could not get travel and task completion time" << std::endl;
		}
	}

	// get the reward for completing the task at the time it would be completed
	double reward = this->children[child_to_evaluate]->task->get_reward_at_time(task_completion_time);

	// get the probability that the task will be completed when I arrive
	int task_index = this->children[child_to_evaluate]->task->get_index();
	double p_complete = this->get_coordinator_probability_for_task(task_index, task_completion_time);

	// return the value of completing the task
	double temp_value = reward*(1 - p_complete) - distance;
	return temp_value;
}

double TreeNode::get_coordinator_probability_for_task(const int &task_index, const double &task_completion_time) {
	// check all of the other robots and determine probability the action will be available at the time I can arrive
	double p_prior = 0.0;
	// check against the human market
	/*if (this->agent->get_index() != -1) {
		double p_complete = world->get_human()->get_coordinator()->get_probability_of_task_completion_at_time(task_index, task_completion_time);
		p_prior = this->probability_update_inclusive(p_prior, p_complete);
	}*/

	// for each robot check the probability of them completing the task before I get there
	for (size_t i = 0; i < world->get_agents().size(); i++) {

		// if its another robot, check against their market
		if (i != this->agent->get_index()) {
			// if I can load their brain
			if (Agent_Coordinator* t_coord = world->get_agents()[i]->get_coordinator()) {
				// update the probability
				double p_complete;
				if (t_coord->get_advertised_task_claim_probability(task_index, task_completion_time, p_complete, world)) {
					p_prior = this->probability_update_inclusive(p_prior, p_complete);
				}
				else {
					std::cerr << "Treenode::get_coordinator_probability_for_task::could not get adverised task claim probability" << std::endl;
				}
			}
		}
	}
	return p_prior;
}

double TreeNode::probability_update_inclusive(const double &a, const double &b) {
	return a + b - a*b;
}

void TreeNode::assemble_market_with_temporal_weighting(Agent_Coordinator * coord, const double & min_prob, int parent_index) {

	// in this use the forecast temporal length to weight the probability of selecting each task
	// that is: P(T_i | t) = f(t, R_i, R_(0->N))
	// as t -> inf : P(T_i | t) -> (R_i - R_min)/(R_max - R_min)
	// as t -> 0 : P(T_i | t) -> 1 if R_i == R_max, 0 else
}

void TreeNode::assemble_market_weighted(Agent_Coordinator* coord, const double &threshold, const int &max_depth, int parent_index) {

	// assign probability to each child and add as stop with time they will be accessed
	for (size_t i = 0; i < this->children.size(); i++) { // 1 because 0 is not a task!
		double p_child_raw = this->children[i]->value / this->sum_child_value;
		this->children[i]->my_probability = this->my_probability * p_child_raw;

		if (this->children[i]->task) {
			int task_index = this->children[i]->task->get_index();
			coord->add_stop_to_my_path(task_index, this->children[i]->my_time, this->children[i]->my_probability);
		}

		// for each child with P > threshold, assemble market on them as well
		if (this->children[i]->my_probability > threshold && this->children[i]->my_depth < max_depth) {
			this->children[i]->assemble_market_weighted(coord, threshold, max_depth, int(i));
		}
	}
}

void TreeNode::assemble_market_greedy(Agent_Coordinator* coord, const int &max_depth, int parent_index) {
	// greedily select best child and add them to market with P(T,t) = 1.0, if best child has task and not below max depth, continue
	if (this->my_depth < max_depth && this->max_child_index >= 0) {
		// my next step is a task
		if (this->children[this->max_child_index]->task) {
			int task_index = this->children[this->max_child_index]->task->get_index();
			coord->add_stop_to_my_path(task_index, this->children[this->max_child_index]->my_time, 1.0);
		}
		// if I am a task
		if (this->task) {
			this->children[this->max_child_index]->assemble_market_greedy(coord, max_depth, this->task->get_index());
		}
		else {
			// it is not a task
			this->children[this->max_child_index]->assemble_market_greedy(coord, max_depth, -1);
		}
	}
}

void TreeNode::monte_carlo_tree_search(const int &max_depth, const double &time_in, std::vector<bool> &task_status, double &max_branch_value) {
	// use mcts to search the tree
	// if I have searched deep enough, stop
	if (this->my_depth > max_depth) {
		return;
	}

	this->nPulls++;

	// if I don't have children, make them
	if (this->children.size() == 0) {
		this->make_children(task_status);
	}

	if (this->children.size() > 0 && this->mcts_find_child(time_in)) {
		// found the MCTS child to search

		// if completing a task was selected then mark it complete
		if (this->children[this->mcts_child_index]->task) {

			// index of task completed
			int task_index = this->children[this->mcts_child_index]->task->get_index();
			task_status[task_index] = false;
		}

		if (this->children[this->mcts_child_index]->value > 0.0) {
			// perform the search
			//UE_LOG(LogTemp, Warning, TEXT("-------------------Searching child %i----------------"), this->mcts_child_index);
			double child_max_value = 0.0;
			this->children[this->mcts_child_index]->monte_carlo_tree_search(max_depth, this->children[this->mcts_child_index]->my_time, task_status, child_max_value);

			// if completing a task was selected then un-mark it
			if (this->children[mcts_child_index]->task) {
				task_status[this->children[this->mcts_child_index]->task->get_index()] = false;
			}
			if (child_max_value > max_branch_value) {
				max_branch_value = child_max_value;
			}
		}
	}
	else {
		//printf("TreeNode::monte carlo tree search: could not find goal child");
	}
}

bool TreeNode::mcts_find_child(const double &time_in) {
	if (world && this->children.size() > 0) { // has tasks and not NULL
												 // initialize vars
		this->max_child_value = -0.0;// -double(INFINITY);
		this->min_child_value = double(INFINITY);
		this->sum_child_value = 0.0;
		this->max_child_index = -1;
		this->mcts_child_index = -1;
		// search through all children and find values + values sum
		for (size_t i = 0; i < this->children.size(); i++) {
			double task_completion_time = 0.0;
			this->children[i]->value = this->child_value_at_time(time_in, int(i), task_completion_time); // get the reward for completing selected task
			this->sum_child_value += std::max(0.0, this->children[i]->value);
			this->children[i]->my_time = task_completion_time;
			if (this->children[i]->value > this->max_child_value) {
				this->max_child_value = std::max(0.0, this->children[i]->value);
				this->max_child_index = int(i);
			}
			if (this->children[i]->value < this->min_child_value) {
				this->min_child_value = std::min(0.0, this->children[i]->value);
			}
		}

		if (this->max_child_value > 0.0) {
			// search through all children for best child
			double max_mcts_reward = 0.0;
			for (size_t i = 0; i < this->children.size(); i++) {
				if (children[i]->value > 0.0) { //  only care about children with value > 0.0
					double pulls_reward = 1.41*sqrt(log(this->nPulls) / (this->children[i]->nPulls + 0.001));
					double value_reward = std::max(0.0, this->children[i]->value) / this->max_child_value;

					if (pulls_reward + value_reward > max_mcts_reward) {
						max_mcts_reward = pulls_reward + value_reward;
						this->mcts_child_index = int(i);
					}
				}
			}
		}

		if (this->mcts_child_index != -1) {
			return true;
		}
		else {
			return false;
		}
	}
	else { // was not properly initialized, return nothing
		std::cerr << "TreeNode::greedy_child_select: not adequately initialized" << std::endl;
		if (this->children.size() == 0) {
			std::cerr << "TreeNode::greedy_child_select: children.size() = " << this->children.size() << std::endl;
		}
		if (!world) {
			std::cerr << "TreeNode::greedy_child_select: world not initialized" << std::endl;
		}
		// error message
		return false;
	}
}

void TreeNode::breadth_first_search(const int &max_depth, const double &time_in, std::vector<bool> &task_status, double &max_branch_value) {
	// if I have searched deep enough, stop
	if (this->my_depth > max_depth) {
		return;
	}

	// if I don't have children, make them
	if (this->children.size() == 0) {
		this->make_children(task_status);
	}

	this->max_child_value = 0.0;// -double(INFINITY);
	this->max_child_index = -1;
	this->sum_child_value = 0.0;
	// search through all children
	for (size_t i = 0; i < this->children.size(); i++) {
		// when was the task completed and what is the value of completing it
		double child_completion_time = 0.0;
		this->children[i]->value = this->child_value_at_time(time_in, int(i), child_completion_time); // get the reward for completing selected task
		this->sum_child_value += std::max(0.0, this->children[i]->value);
		this->children[i]->my_time = child_completion_time;

		// print out a status message for debugging
		if (this->my_depth == 0) {
			//UE_LOG(LogTemp, Warning, TEXT("TreeNode::breadth_first_search: depth: %i; my_children[%i]->value: %f"), this->my_depth, i, this->children[i]->value);
		}
		else if (this->task) {
			//UE_LOG(LogTemp, Warning, TEXT("TreeNode::breadth_first_search: depth: %i; my_task: %i, my_children[%i]->value: %f"), this->my_depth, this->task->get_index(), i, this->children[i]->value);
		}
		if (this->children[i]->value > this->max_child_value) {
			this->max_child_value = this->children[i]->value;
			this->max_child_index = int(i);
		}

		// if completing a task was selected then mark it complete
		if (this->children[i]->task) {
			// index of task completed
			int task_index = this->children[i]->task->get_index();
			task_status[task_index] = false;
		}

		double temp_val = 0.0;
		this->children[i]->breadth_first_search(max_depth, this->children[i]->my_time, task_status, temp_val);

		// if completing a task was selected then mark it incomplete
		if (this->children[i]->task) {
			// index of task completed
			int task_index = this->children[i]->task->get_index();
			task_status[task_index] = true;
		}
		if (temp_val > max_child_value) {
			max_child_value = temp_val;
		}
	}
	max_branch_value = this->value + max_child_value;
}

void TreeNode::mcts_rollout() {

}

// greedy search up through a specified depth
void TreeNode::greedy_search(const int &max_depth, const double &time_in, std::vector<bool> &task_status) {
	// if I have searched deep enough, stop
	if (this->my_depth > max_depth) {
		return;
	}

	// if I don't have children, make them
	if (this->children.size() == 0) {
		this->make_children(task_status);
	}

	// if I am able to find a best child
	if (this->children.size() > 0 && this->greedy_find_child(time_in)) {

		// if completing a task was selected then mark it complete
		if (this->children[this->max_child_index]->task) {
			// index of task completed
			int task_index = this->children[this->max_child_index]->task->get_index();

			task_status[task_index] = false;
		}

		if (this->children[this->max_child_index]->value > 0.0) {
			// perform the search
			//UE_LOG(LogTemp, Warning, TEXT("-------------------Searching child %i----------------"), this->max_child_index);
			this->children[this->max_child_index]->greedy_search(max_depth, this->children[this->max_child_index]->my_time, task_status);

			// if completing a task was selected then un-mark it
			if (this->children[max_child_index]->task) {
				task_status[this->children[this->max_child_index]->task->get_index()] = false;
			}
		}
	}
	else {
		//UE_LOG(LogTemp, Warning, TEXT("TreeNode::greedy_search: could not find goal child" << std::endl;
	}
}

// greedy search up through a specified depth
void TreeNode::greedy_search(const int &max_depth, const double &time_in, std::vector<int> &available_tasks) {
	// if I have searched deep enough, stop
	if (this->my_depth > max_depth) {
		return;
	}

	// if I don't have children, make them
	if (this->children.size() == 0) {
		this->make_children(available_tasks);
	}

	// if I am able to find a best child
	if (this->children.size() > 0 && this->greedy_find_child(time_in)) {

		// if completing a task was selected then mark it complete
		if (this->children[this->max_child_index]->task && this->children[this->max_child_index]->value > 0.0) {
			// index of task completed
			int task_index = this->children[this->max_child_index]->task->get_index();

			// remove from available tasks
			int at_index = -1;
			for (size_t i = 0; i < available_tasks.size(); i++) {
				if (available_tasks[i] == task_index) {
					at_index = i;
					break;
				}
			}
			available_tasks.erase(available_tasks.begin() + at_index);

			// perform the search
			//printf("-------------------Searching child %i----------------"), this->max_child_index);
			this->children[this->max_child_index]->greedy_search(max_depth, this->children[this->max_child_index]->my_time, available_tasks);

			// if completing a task was selected then un-mark it
			available_tasks.insert(available_tasks.begin() + at_index, task_index);
		}
	}
	else {
		//printf("TreeNode::greedy_search: could not find goal child");
	}
}

bool TreeNode::greedy_find_child(const double &time_in) {
	if (world && this->children.size() > 0) { // has tasks and not NULL
												 // initialize vars
		this->max_child_value = 0.0;// -double(INFINITY);
		this->max_child_index = -1;
		this->sum_child_value = 0.0;
		// search through all children
		for (size_t i = 0; i < this->children.size(); i++) {
			double child_completion_time = 0.0;
			this->children[i]->value = this->child_value_at_time(time_in, int(i), child_completion_time); // get the reward for completing selected task
			this->sum_child_value += std::max(0.0, this->children[i]->value);
			this->children[i]->my_time = child_completion_time;
			// print out a status message for debugging
			if (this->my_depth == 0) {
				//UE_LOG(LogTemp, Warning, TEXT("TreeNode::greedy_search: depth: %i; my_children[%i]->value: %f"), this->my_depth, i, this->children[i]->value);
			}
			else if (this->task) {
				//UE_LOG(LogTemp, Warning, TEXT("TreeNode::greedy_search: depth: %i; my_task: %i, my_children[%i]->value: %f"), this->my_depth, this->task->get_index(), i, this->children[i]->value);
			}
			if (this->children[i]->value > this->max_child_value) {
				this->max_child_value = this->children[i]->value;
				this->max_child_index = int(i);
			}
		}
		if (max_child_index != -1) {
			return true;
		}
		else {
			return false;
		}
	}
	else { // was not properly initialized, return nothing
		std::cerr << "TreeNode::greedy_child_select: not adequately initialized" << std::endl;
		if (this->children.size() == 0) {
			std::cerr << "TreeNode::greedy_child_select: children.size() = " << this->children.size() << std::endl;
		}
		if (!world) {
			std::cerr << "TreeNode::greedy_child_select: world not initialized" << std::endl;
		}
		// error message
		return false;
	}
}

void TreeNode::exploit_tree(const int &max_depth, std::vector<int> &path) {

	if (this->my_depth == 0) {
		path.push_back(this->agent->get_loc());
	}

	if (this->my_depth > max_depth) {
		return;
	}

	int maxdex = -1;
	double max_value = -double(INFINITY);
	for (size_t i = 0; i < this->children.size(); i++) {
		if (this->children[i]->value > max_value) {
			max_value = this->children[i]->value;
			maxdex = int(i);
		}
	}

	if (maxdex != -1) {
		if (this->children[maxdex]->task) {
			path.push_back(this->children[maxdex]->task->get_index());
			this->children[maxdex]->exploit_tree(max_depth, path);
		}
	}
	else {
		return;
	}
}