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
	Distributed_MCTS(World* world, Map_Node* task_in, Agent* agent_in, Distributed_MCTS* parent, const int &my_kid_index, const double &parent_time_in, const int update_index);
	~Distributed_MCTS();

	Agent* get_agent() { return this->agent; };
	double get_alpha() { return this->alpha; };
	double get_down_branch_expected_reward() {return this->down_branch_expected_reward; }; // might have to calc a few things, not a simple return
	double get_distance() {return this->distance; };
	double get_expected_reward() {return this->expected_reward; };
	double get_n_pulls() { return this->number_pulls; };
	double get_raw_probability() { return this->raw_probability; };
	void set_raw_probability(const double &rp){this->raw_probability = rp; };
	double get_branch_probability() {return this->branch_probability; };
	void set_branch_probability(const double &bp) {this->branch_probability = bp; };
	double get_raw_reward() { return this->raw_reward; };
	std::vector<Distributed_MCTS*> get_kids() {return this->kids; }; // get access to kids
	Map_Node* get_task() { return this->task; };
	int get_task_index() { return this->task_index; };
	double get_completion_time() { return this->completion_time; };
	void set_task_index(const int &ti) {this->task_index=ti; };
	void reset_mcts_team_prob_actions();
	void set_as_root() {this->raw_probability = 1.0; };

	// call from parent not self
	void search(const bool &am_root, const int &depth_in, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, int &rollout_depth, const int &update_index);
	void reset_task_availability_probability() { this->probability_task_available = -1.0; };
	void update_probable_actions();
	void sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in); // get my probabilities to advertise
	bool exploit_tree(int &goal_index, std::vector<std::string> &args, std::vector<double> &vals);
	void prune_branches(const int &max_child);
	void burn_branches(); // don't save any kids, burn it all
	void get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards);

private:
	// rollout does not create new nodes
	bool make_kids(const std::vector<bool> &task_status, const std::vector<int> &task_set, const int &update_index);
	bool ucb(Distributed_MCTS* &gc);
	// update line for private
	void update_down_branch_expected_reward(Distributed_MCTS* &gc); // update my branch reward, min, max, and sum
	void update_down_branch_expected_reward(); // update my branch reward, min, max, and sum
	void find_kid_probabilities(); // find and assign my kid probabilities
	void update_raw_probability();
	void update_my_completion_time();
	void perform_initial_sampling();

	Agent* agent;
	Map_Node* task;
	int task_index;
	World* world;
	Distributed_MCTS* parent;
	int last_update_index;
	
	double alpha; // gradient descent reward
	int max_rollout_depth, max_search_depth; // how far do I search
	double probability_task_available;
	double distance; // how far from parent am I by astar?
	double travel_time; // time by astar path?
	double work_time; // once there, how long to complete?
	double completion_time; // time I actually finish and am ready to leave
	double parent_time; // when should I calc travel time from
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

	void add_sw_uct_update(const double &min, const double &max, const int &planning_iter);

	double number_pulls; // how many times have I been pulled
	double beta, epsilon, gamma; // for ucb, d-ducb, sw-ucb
	std::vector<Distributed_MCTS*> kids; // my kids


};


