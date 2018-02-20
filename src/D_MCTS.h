#pragma once

#include <vector>
#include <string>

class Map_Node;
class Agent;
class Agent_Coordinator;
class World;

class D_MCTS
{
public:
	D_MCTS(World* world, Map_Node* task_in, Agent* agent_in, D_MCTS* parent, const int &my_kid_index, const double &parent_time_in);
	~D_MCTS();

	Agent* get_agent() { return this->agent; };
	double get_alpha() { return this->alpha; };
	double get_branch_reward(); // might have to calc a few things, not a simple return
	double get_distance() {return this->distance; };
	double get_expected_reward(); // might have to calc some things, not a simple return
	double get_n_pulls() { return this->number_pulls; };
	double get_probability() { return this->probability; };
	double get_reward() { return this->reward; };
	std::vector<D_MCTS*> get_kids() {return this->kids; }; // get access to kids
	Map_Node* get_task() { return this->task; };
	int get_task_index() { return this->task_index; };
	double get_completion_time() { return this->completion_time; };
	void set_task_index(const int &ti);
	void reset_mcts_team_prob_actions();
	void set_as_root() {this->probability = 1.0; };

	// call from parent not self
	void search_from_root(std::vector<bool> &task_status, std::vector<int> &task_set);
	void search(const int &depth_in, const double &time_in, std::vector<bool> &task_status, std::vector<int> &task_set, int &rollout_depth);
	void reset_task_availability_probability() { this->probability_task_available = -1.0; };
	void set_probability(const double &prob){ this->probability = prob; };
	void update_probable_actions();
	void sample_tree_and_advertise_task_probabilities(Agent_Coordinator* coord_in); // get my probabilities to advertise
	bool exploit_tree(int &goal_index, std::vector<std::string> &args, std::vector<double> &vals);
	void prune_branches(const int &max_child);
	void burn_branches(); // don't save any kids, burn it all
	void get_best_path(std::vector<int> &path, std::vector<double> &times, std::vector<double> &rewards);

private:
	// rollout does not create new nodes
	bool make_kids(std::vector<bool> &task_status, std::vector<int> &task_set);
	void update_max_branch_reward_kid(D_MCTS* gc); // check if I need to update max kid and update it and my branch reward if I have to
	void update_min_branch_reward_kid(D_MCTS* gc); // check if I need to update max kid and update if I have to
	void update_kid_rewards_with_new_probabilities();
	void update_branch_reward(); // update my branch reward, min, max, and sum
	bool ucb_select_kid(D_MCTS* &gc);
	void find_kid_probabilities(); // find and assign my kid probabilities

	Agent* agent;
	Map_Node* task;
	int task_index;
	World* world;
	D_MCTS* parent;
	double last_update_time;
	
	double alpha; // gradient descent reward
	int max_rollout_depth, max_search_depth;
	double probability_task_available;
	double distance; // how far from parent am I by astar?
	double travel_time; // time by astar path?
	double work_time; // once there, how long to complete?
	double completion_time; // time I actually finish and am ready to leave
	double parent_time; // when should I calc travel time from
	double reward; // my reward at e_time/time?
	double expected_reward; // my expected reward for completing task at e_time/time with e_dist/dist?
	double reward_weighting, distance_weighting; // for reward function
	double branch_reward; // my and all my best kids expected reward combined
	int last_planning_iter_end; // what was the end of the last iter called, to know if i need to resample the probabilities
	double wait_time; // how long do I wait if I select a null action

	std::string search_type;
	std::vector<int> time_log; // record keeping
	std::vector<double> exp_reward_log; // record keeping
	double sum_exp_reward, sum_n_pulls;
	int window_width;

	int max_kid_index; // current golden child
	double max_kid_branch_reward; // their reward, for normalizing

	double probability; // probability of me being selected, function of my relative reward to adjacent kids and parent's probability
	double sampling_probability_threshold; // when probability drops below this, stop searching

	void add_sw_uct_update(const double &min, const double &max, const int &planning_iter);

	double number_pulls; // how many times have I been pulled
	double beta, epsilon, gamma; // for ucb, d-ducb, sw-ucb
	std::vector<D_MCTS*> kids; // my kids
};


