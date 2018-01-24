#pragma once

#include <vector>
#include <string>

class Map_Node;
class Agent;
class Agent_Coordinator;
class World;

class MCTS
{
public:
	MCTS(World* world, Map_Node* task_in, Agent* agent_in, MCTS* parent, const int &my_kid_index, const double &parent_time_in);
	~MCTS();

	Agent* get_agent() { return this->agent; };
	double get_branch_value(); // might have to calc a few things, not a simple return
	double get_distance() {return this->distance; };
	double get_expected_value(); // might have to calc some things, not a simple return
	double get_search_value(const double &min, const double &max, const int &pull_iter); // might have to calc values, not a simple return
	int get_kid_index() { return this->kid_index; }
	double get_max_kid_distance_threshold() { return this->max_kid_distance_threshold; };
	double get_n_pulls() { return this->number_pulls; };
	double get_probability() { return this->probability; };
	double get_reward() { return this->reward; };
	Map_Node* get_task() { return this->task; };
	int get_task_index() { return this->task_index; };
	double get_completion_time() { return this->completion_time; };
	void set_task_index(const int &ti);
	void reset_mcts_team_prob_actions();

	// call from parent not self
	void search_from_root(std::vector<bool> &task_status, std::vector<int> &task_set, const int &last_planning_iter_end, const int &planning_iter);
	void search(const int &depth_in, double &passed_reward, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, const int &last_planning_iter_end, const int &planning_iter);
	bool kid_pruning_heuristic(const std::vector<bool> &task_status);
	void reset_task_availability_probability() { this->probability_task_available = -1.0; };
	void set_probability(const double &sum_value, const double &parent_probability);
	void set_probability(const double &sum, const double &min_value, const double &max_value, const double &parent_probability);
	void set_as_root();
	void sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in); // get my probabilities to advertise
	bool exploit_tree(int &goal_index, std::vector<std::string> &args, std::vector<double> &vals);
	void prune_branches(); // if I just moved on to the next branch then I should prune old branches
	void burn_branches(); // don't save any kids, burn it all
	MCTS* get_golden_child(); // get pointer to golden child, mainly for moving to next branch
	void get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards);

private:
	// rollout does not create new nodes
	void rollout(const int &c_index, const int &rollout_depth, const double&time_in, std::vector<bool> &task_status, std::vector<int> &task_set, double &passed_value);
	bool make_kids(std::vector<bool> &task_status, std::vector<int> &task_set);
	bool make_nbr_kids(const std::vector<bool> &task_status);
	void keep_n_best_kids(MCTS* kiddo);
	void find_max_branch_value_kid(); // find the maximum expected value kid
	void find_min_branch_value_kid(); // find the minimum expected value kid 
	void find_sum_kid_branch_value(); // find the sum of all of my kids branch values
	void update_max_branch_value_kid(MCTS* gc); // check if I need to update max kid and update it and my branch value if I have to
	void update_min_branch_value_kid(MCTS* gc); // check if I need to update max kid and update if I have to
	void update_kid_values_with_new_probabilities();
	void update_branch_values(MCTS* gc, const double &kids_prior_branch_value); // update my branch value, min, max, and sum
	void erase_null_kids();

	bool find_kid_to_search(const std::vector<bool> &task_status, MCTS* &gc, const int &search_iter);
	void find_kid_probabilities(); // find and assign my kid probabilities

	Agent* agent;
	Map_Node* task;
	int task_index;
	World* world;
	MCTS* parent;
	double last_update_time;
	
	int max_rollout_depth, max_search_depth;
	double rollout_reward;
	double max_kid_distance_threshold; // how far can a node be and still be a child
	double probability_task_available;
	double distance; // how far from parent am I by astar?
	double travel_time; // time by astar path?
	double work_time; // once there, how long to complete?
	double completion_time; // time I actually finish and am ready to leave
	double parent_time; // when should I calc travel time from
	double reward; // my reward at e_time/time?
	double expected_value; // my expected value for completing task at e_time/time with e_dist/dist?
	double reward_weighting, distance_weighting; // for value function
	double branch_value; // my and all my best kids expected value combined
	double exploit_value; // weighted expected value
	double explore_value; // value of searching rare arms
	double search_value; // exploit value + explore value
	int last_planning_iter_end; // what was the end of the last iter called, to know if i need to resample the probabilities
	double wait_time; // how long do I wait if I select a null action

	std::string search_type;
	std::vector<int> time_log; // record keeping
	std::vector<double> exp_value_log; // record keeping
	double sum_exp_value, sum_n_pulls;
	int window_width;

	int kid_index; // my index in my parents kids
	int max_kid_index, min_kid_index; // current golden child
	double max_kid_branch_value, min_kid_branch_value, sum_kid_branch_value; // their value, for normalizing

	double probability; // probability of me being selected, function of my relative value to adjacent kids and parent's probability
	double sampling_probability_threshold; // when probability drops below this, stop searching

	void add_sw_uct_update(const double &min, const double &max, const int &planning_iter);

	double number_pulls; // how many times have I been pulled
	double beta, epsilon, gamma; // for ucb, d-ducb, sw-ucb
	std::vector<MCTS*> kids; // my kids
};


