#pragma once

#include <vector>
#include <string>

class Probability_Node;
class World;
class Agent;

class Agent_Coordinator
{
public:
	Agent_Coordinator(Agent* agent, int n_nodes);
	~Agent_Coordinator();

	Agent* agent;
	std::vector<Probability_Node*> get_prob_actions() { return this->prob_actions; };
	// coordination stuff
	bool get_advertised_task_claim_probability(const int &task_num, const double &query_time, double &prob_taken, World* world);
	bool get_claims_after(const int &task, const double &query_time, std::vector<double>& prob, std::vector<double>& times);
	bool advertise_task_claim(World* world);
	void reset_prob_actions(); // reset my probable actions
	void add_stop_to_my_path(const int &task_index, const double &time, const double &prob);
	double get_reward_impact(const int &task, const int &agent, const double &completion_time, World* world);
	void print_prob_actions();
	void get_plan( std::vector<int> &claimed_tasks, std::vector<double> &claimed_time, std::vector<double> &claimed_probability);
	void upload_new_plan(const std::vector<int> &claimed_tasks, const std::vector<double> &claimed_time, const std::vector<double> &claimed_probability);

private:
	std::vector<Probability_Node*> prob_actions;
	int n_tasks;

	std::string task_claim_time, task_claim_method;
};

