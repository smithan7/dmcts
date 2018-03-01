#pragma once

#include <vector>
#include <string>

class Map_Node;
class Agent;
class Agent_Coordinator;
class World;

class Distributed_MCTS
{
public:
	Distributed_MCTS(World* world, Map_Node* task_in, Agent* agent_in, Distributed_MCTS* parent, const int update_index);
	~Distributed_MCTS();

	Agent* get_agent() { return this->agent; };
	double get_alpha() { return this->alpha; };
	double get_down_branch_expected_reward() {return this->down_branch_expected_reward; }; // might have to calc a few things, not a simple return
	double get_cumulative_reward() {return this->cumulative_reward; };
	double get_distance() {return this->distance; };
	double get_expected_reward() {return this->expected_reward; };
	double get_n_pulls() { return this->number_pulls; };
	double get_raw_probability() { return this->raw_probability; };
	double get_branch_probability() {return this->branch_probability; };
	double get_raw_reward() { return this->raw_reward; };
	int get_task_index() { return this->task_index; };
	double get_completion_time() { return this->completion_time; };
	std::vector<Distributed_MCTS*> get_kids() {return this->kids; }; // get access to kids
	Map_Node* get_task() { return this->task; };
	
	void set_raw_probability(const double &rp){this->raw_probability = rp; };
	void set_branch_probability(const double &bp) {this->branch_probability = bp; };
	void set_cumulative_reward(const double &cr) {this->cumulative_reward = cr; };
	void set_task_index(const int &ti) {this->task_index=ti; };
	void set_as_root() {this->raw_probability = 1.0; this->parent = NULL; };

	// call from parent not self
	void search(const int &max_search_depth_in, int depth_in, Distributed_MCTS* parent, std::vector<bool> &task_status, std::vector<int> &task_set, int rollout_depth, const int &update_index);
	void sample_tree(Agent_Coordinator* coord_in, int &depth, const int &update_index);
	void sample_tree(int &depth);
	bool exploit_tree(int &goal_index, std::vector<std::string> &args, std::vector<double> &vals, int depth, const int &update_index);
	void get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards, int depth, const int &update_index);

	// get rid of old parts of tree
	void prune_branches(const int &max_child); // get rid of all kids except my best
	void burn_branches(); // don't save any kids, burn it all
	void clean_null_branches(int depth); // Get rid of all branches containing null tasks

private:
	// rollout does not create new nodes
	bool make_kids(const std::vector<bool> &task_status, const std::vector<int> &task_set, const int &update_index);
	bool ucb(Distributed_MCTS* &gc);
	void check_completion_time(const int &depth, const int &update_index); // Check that I will complete this task at the advertised time

	// update line for private
	void update_down_branch_expected_reward(Distributed_MCTS* &gc); // update my branch reward, min, max, and sum
	void update_down_branch_expected_reward(); // update my branch reward, min, max, and sum
	void update_probability_task_is_available();
	void update_my_completion_time();
	void update_my_travel_time();
	void perform_initial_sampling();

	Agent* agent;
	Map_Node* task;
	int task_index;
	World* world;
	Distributed_MCTS* parent; // who is my parent?
	int parent_index; // what node is my parent at?
	double parent_time; // when is my parent done with their task?

	bool use_impact; // am I strictly reward or do I use impact?


	int last_update_index; // When was the last time I updated my coordination
	
	double alpha; // gradient descent reward
	int max_rollout_depth, max_search_depth; // how far do I search
	double probability_task_available;
	double distance; // how far from parent am I by astar?
	double travel_time; // time by astar path?
	double work_time; // once there, how long to complete?
	double completion_time; // time I actually finish and am ready to leave
	double raw_reward; // my reward at e_time/time?
	double expected_reward; // my expected reward for completing task at e_time/time with e_dist/dist?
	double down_branch_expected_reward; // my and all my best kids expected reward combined
	double cumulative_reward; // total reward for all pulls
	double mean_reward; // cumulative_reward / n_pulls
	int last_planning_iter_end; // what was the end of the last iter called, to know if i need to resample the probabilities
	
	std::string search_type;
	std::vector<int> time_log; // record keeping
	std::vector<double> exp_reward_log; // record keeping
	double sum_exp_reward, sum_n_pulls;
	int window_width;

	double raw_probability; // probability of me being selected, function of my relative reward to adjacent kids
	double branch_probability; // How likely am I to be selected accounting for my parent
	double min_sampling_probability_threshold; // when probability drops below this, stop searching

	double number_pulls; // how many times have I been pulled
	double beta, epsilon, gamma; // for ucb, d-ducb, sw-ucb
	std::vector<Distributed_MCTS*> kids; // my kids


};