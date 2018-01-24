#pragma once

#include <vector>

class Probability_Node
{
public:
	// constructor
	Probability_Node(int i);
	Probability_Node(std::vector<double>& p_in, std::vector<double>& t_in, int ti_in);

	// destructor
	~Probability_Node();

	// update using a new nodes
	void update(Probability_Node* msg);

	// insert a new possible arrival time and update following arrival times
	void add_stop_to_my_path(const double &time, const double &prob);

	// insert a new possible arrival time and update following arrival times
	void add_stop_to_shared_path(const double &time, const double &prob);

	// update probability, a + b - a*b
	double probability_update_inclusive(const double &plan, const double &msg);

	// update probability for exclusive events, a + b
	double probability_update_exclusive(const double &plan, const double &msg);

	// update probability by removing the msg
	double probability_removal_inclusive(const double & plan, const double & rem);

	// update probability by removing the msg
	double probability_removal_exclusive(const double & plan, const double & rem);

	// return the probability at time specified
	double get_probability_at_time(const double &time);

	// clear all stops except 0, 0.0 and inf, 1.0
	void reset();

	// print out the probability table
	void print_out();

	// return vector containing all probabilities of arrival
	std::vector<double>& get_probability_of_completion() { return this->probability_of_completion; };

	// return vector containing all arrival times
	std::vector<double>& get_completion_time() { return this->completion_time; }

	// return single int giving task index correspoinding to this node
	int get_task_index() { return this->task_index; }

	// have i been claimed? do I need to be reset or searched over?
	bool claimed();

	// if there are claims after the query time, add to list and return true, exclude inf
	bool get_claims_after(double query_time, std::vector<double> &probs, std::vector<double> &times);

	// get all claims, used only for ROS
	void get_claims(std::vector<int> &indices, std::vector<double> &times, std::vector<double> &probs);

	// load all new claims, used only for ROS
	void insert_claim(const double &claimed_time, const double &claimed_probability);

private:
	// when will it be completed
	std::vector<double> completion_time;
	// what is the probability it will be completed
	std::vector<double> probability_of_completion;
	// when was the report made about it being completed
	std::vector<int> report_times;
	// who made the report
	std::vector<int> reporting_agent;

	int task_index;

	double getCDF(double x);
	double getPDF(double x);
	double mean, stan_dev;
	double pi, sqrt_pi;
};

