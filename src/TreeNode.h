#pragma once

#include <vector>

class World;
class Agent;
class Map_Node;
class Agent_Coordinator;

class TreeNode
{
public:
	// constructor
	TreeNode(World* world, Map_Node* task, Agent* agent, int depth_in);

	// destructor
	~TreeNode();

	// value of going to me
	double value;

	// number of times I have been searched
	double nPulls;

	// maximum value of my children
	double max_child_value;

	// minimum value of my children
	double min_child_value;

	// sum of all children, used to calculate the market
	double sum_child_value;

	// index to child with max value
	int max_child_index;

	// which child is mcts going to search next
	int mcts_child_index;

	// my actor, may be a robot or a human
	Agent* agent;
	
	// my task that this node exists on, NULL for root which is on robot
	Map_Node* task;

	// my children
	std::vector<TreeNode*> children;

	// master linker
	World* world;

	// depth of the tree search
	int my_depth;

	// how likely am I to be selected
	double my_probability;

	// what time am I searching from?
	double my_time;

	// how long do I wait if doing nothing
	double wait_time;

	// greedy search up through a specified depth
	void greedy_search(const int &max_depth, const double &initial_time, std::vector<bool> &task_status);

	// greedy search up through a specified depth
	void greedy_search(const int &max_depth, const double &initial_time, std::vector<int> &available_tasks);


	// breadth first search through depth
	void breadth_first_search(const int &max_depth, const double &time_in, std::vector<bool> &task_status, double &max_branch_value);

	// monte carlo tree search
	void monte_carlo_tree_search(const int &max_depth, const double &time_in, std::vector<bool> &task_status, double &max_branch_value);

	// rollout for monte carlo tree search
	void mcts_rollout();

	// exploit tree following search and get my likely path
	void exploit_tree(const int &max_depth, std::vector<int> &path);

	// draw my path
	void draw_my_path(const std::vector<int> &path);

	// state of world at time
	double evaluate_world_at_time(const double &time);

	// state of world except specified task (assume ut was completed)
	bool evaluate_at_time_except(const double &initial_time, const int &index_to_evaluate, double &task_completion_time, double &penalty);

	// value of completing the specific task, reward(t)_i - distance
	double child_value_at_time(const double &initial_time, const int &index_to_evaluate, double &task_completion_time);

	// check against all other robot markets and determine the probability that each task will be taken
	double get_coordinator_probability_for_task(const int &task_index, const double &task_completion_time);

	// initialize all of my children
	void make_children(const std::vector<bool> &task_status);

	// initialize all of my children
	void make_children(const std::vector<int> &available_tasks);

	// find the best child and assign their values, times, and my max child value
	bool greedy_find_child(const double &time_in);

	// find the next child to search via mcts
	bool mcts_find_child(const double &time_in);

	// how long to travel from one location to given task and complete it
	bool get_travel_and_task_completion_time(const int &start_node, const int &goal_node, const double &time_in, double &distance, double &completion_time);

	// update multiple path probability
	double probability_update_inclusive(const double &a, const double &b);

	// assemble the market by sampling the tree
	void assemble_market_weighted(Agent_Coordinator* coord, const double &threshold, const int &max_depth, int parent_index);

	// assemble the market greedily
	void assemble_market_greedy(Agent_Coordinator* coord, const int &max_depth, int parent_index);

	// as I plan further into the future reduce the certainty of choosing the best task
	void assemble_market_with_temporal_weighting(Agent_Coordinator* coord, const double &min_prob, int parent_index);
};
