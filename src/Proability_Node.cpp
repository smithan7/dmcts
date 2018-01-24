#include "Proabability_Node.h"

#include <algorithm>


Probability_Node::Probability_Node(int i) {
	this->task_index = i;

	// at t=0, prob of completing = 0
	this->probability_of_completion.push_back(0.0);
	this->completion_time.push_back(0.0);

	// at t=inf, prob of completing = 1.0
	this->probability_of_completion.push_back(1.0);
	this->completion_time.push_back(double(INFINITY));

	this->pi = 3.14159265358979323846264338327950288419716939937510582;
	this->sqrt_pi = sqrt(pi);
}

Probability_Node::Probability_Node(std::vector<double> &p_in, std::vector<double> &t_in, int ti_in) {
	this->completion_time = t_in;
	this->probability_of_completion = p_in;
	this->task_index = ti_in;
}

void Probability_Node::update(Probability_Node* msg) {
	// there are two ways to do this
		// 1- reporting agents is all agents, 1-n_agents and time is the reporting time, -1 if not heard from. Much easier in search but takes extra memory
		// 2- reporting agents is only agents that I have heard from, slower in search cheaper in memory.

	for (size_t a = 0; a < msg->reporting_agent.size(); a++) {
		// if they have a newer reporting time than me, remove my points for that agent
		if (msg->reporting_agent[a]) {

		}
	}
}

bool Probability_Node::claimed() {
	if (this->completion_time.size() > 2 || this->probability_of_completion.size() > 2) {
		return true;
	}
	else {
		return false;
	}
}

void Probability_Node::get_claims(std::vector<int> &indices, std::vector<double> &times, std::vector<double> &probs){
	if (this->probability_of_completion.size() == 2) {
		return;
	}
	else {
		for (int i = 1; i < this->probability_of_completion.size() - 1; i++) {
			indices.push_back(this->task_index);
			probs.push_back(this->probability_of_completion[i]);
			times.push_back(this->completion_time[i]);
		}
	}
}

void Probability_Node::insert_claim(const double &time, const double &prob){
		int time_index = -1;
	for (size_t i = 1; i < this->completion_time.size(); i++) { // find place to insert the new point so that path is in temporal order
		if (time > this->completion_time[i - 1] && time <= this->completion_time[i] && this->probability_of_completion[i-1] < 1.0) { // needs to be lower than the next one not larger
			time_index = int(i);
			break;
		}
	}
	if (time_index >= 0) { // was a point found?
						   // yes, insert at the found point
		if (this->completion_time[time_index] == time) { // is it at the same time?
			// yes, replace existing probability
			this->probability_of_completion[time_index] = prob;
			return;
		}
		else {
			// no, insert the new time
			this->probability_of_completion.insert(this->probability_of_completion.begin() + time_index, prob);
			this->completion_time.insert(this->completion_time.begin() + time_index, time);
		}
	}
}

bool Probability_Node::get_claims_after(double query_time, std::vector<double>& probs, std::vector<double>& times){
	// if there are claims after the query time, add to list and return true, exclude inf
	if (this->probability_of_completion.size() == 2) {
		return false;
	}
	else {
		probs.clear();
		times.clear();
		for (int i = 1; i < this->probability_of_completion.size() - 1; i++) {
			if (this->completion_time[i] > query_time) {
				probs.push_back(this->probability_of_completion[i] - this->probability_of_completion[i-1]);
				times.push_back(this->completion_time[i]);
			}
		}
		if (probs.size() > 0) {
			return true;
		}
		else {
			return false;
		}

	}
}

Probability_Node::~Probability_Node() {}


void Probability_Node::add_stop_to_my_path(const double &time, const double &prob) {

	double p = prob;
	if (p <= 0) {
		return;
	}
	else {
		p = std::min(prob, 1.0);
	}


	//UE_LOG(LogTemp, Error, TEXT("Probability_Node::add_stop_to_my_path: completion_time.size(): %i"), this->completion_time.size());
	int time_index = -1;
	for (size_t i = 1; i < this->completion_time.size(); i++) { // find place to insert the new point so that path is in temporal order
		if (time > this->completion_time[i - 1] && time <= this->completion_time[i] && this->probability_of_completion[i-1] < 1.0) { // needs to be lower than the next one not larger
			time_index = int(i);
			break;
		}
	}
	if (time_index >= 0) { // was a point found?
						   // yes, insert at the found point
		if (this->completion_time[time_index] == time) { // is it at the same time?
													  // yes, merge the two probabilities
			this->probability_of_completion[time_index] = this->probability_update_exclusive(p, this->probability_of_completion[time_index]);
		}
		else {
			// no, insert the new time
			double updated_prob = this->probability_update_exclusive(p, this->probability_of_completion[time_index - 1]);
			this->probability_of_completion.insert(this->probability_of_completion.begin() + time_index, updated_prob);
			this->completion_time.insert(this->completion_time.begin() + time_index, time);

		}
		// update all following times
		for (size_t i = size_t(time_index) + 1; i < this->probability_of_completion.size(); i++) {
			this->probability_of_completion[i] = this->probability_update_exclusive(p, this->probability_of_completion[i]);
		}
	}
}

void Probability_Node::add_stop_to_shared_path(const double &time, const double &prob) {

	//UE_LOG(LogTemp, Error, TEXT("Probability_Node::add_stop_to_my_path: completion_time.size(): %i"), this->completion_time.size());
	int time_index = -1;
	for (size_t i = 1; i < this->completion_time.size(); i++) { // find place to insert the new point so that path is in temporal order
															 //UE_LOG(LogTemp, Error, TEXT("%0.2f > %0.2f && %0.2f <= %0.2f = %i"), time, this->completion_time[i - 1], time, this->completion_time[i], time > this->completion_time[i - 1] && time <= this->completion_time[i]);
		if (time > this->completion_time[i - 1] && time <= this->completion_time[i]) { // needs to be lower than the next one not larger
																				 //UE_LOG(LogTemp, Error, TEXT("Probability_Node::add_stop_to_my_path: %i"), i);
			time_index = int(i);
			break;
		}
	}
	if (time_index >= 0) { // was a point found?
						   // yes, insert at the found point
		if (this->completion_time[time_index] == time) { // is it at the same time?
													  // yes, merge the two probabilities
			this->probability_of_completion[time_index] = this->probability_update_inclusive(prob, this->probability_of_completion[time_index]);
		}
		else {
			// no, insert the new time
			double updated_prob = this->probability_update_inclusive(prob, this->probability_of_completion[time_index]);
			this->probability_of_completion.insert(this->probability_of_completion.begin() + time_index, updated_prob);
			this->completion_time.insert(this->completion_time.begin() + time_index, time);

		}
		// update all following times
		for (size_t i = size_t(time_index) + 1; i < this->probability_of_completion.size(); i++) {
			this->probability_of_completion[i] = this->probability_update_inclusive(prob, this->probability_of_completion[i]);
		}
	}
}

double Probability_Node::get_probability_at_time(const double &time) {
	if (this->completion_time.size() == this->probability_of_completion.size() && this->probability_of_completion.size() != 2) { // ensure they are the same size
		for (size_t i = 1; i < this->completion_time.size(); i++) {
			if (time < this->completion_time[i] && time >= this->completion_time[i - 1]) {
				return this->probability_of_completion[i - 1]; // view as a series of step functions, which step am I on?
			}
		}
	}
	return 0.0f;
}

double Probability_Node::probability_update_inclusive(const double &plan, const double &msg) {
	return plan + msg - plan*msg;
}

double Probability_Node::probability_update_exclusive(const double &plan, const double &msg) {
	if (1.0f > plan + msg) {
		return plan + msg;
	}
	else {
		return 1.0f;
	}
}

double Probability_Node::probability_removal_inclusive(const double &plan, const double &rem) {
	if (rem >= 1.0) {
		return 0.0;
	}
	if (rem <= 0.0) {
		return plan;
	}
	return (rem - plan)/(rem - 1.0);
}

double Probability_Node::probability_removal_exclusive(const double &plan, const double &rem) {
	double val = plan - rem;
	if (val >= 1.0) {
		return 1.0;
	}
	if (val <= 0.0) {
		return 0.0;
	}
	return val;
}

double Probability_Node::getPDF(double x) {
	double pdf = 1 / sqrt(2 * pow(this->stan_dev, 2)*pi)*exp(-pow(x - this->mean, 2) / (2 * pow(this->stan_dev, 2)));
	return pdf;
}

double Probability_Node::getCDF(double x) {
	// A Sigmoid Approximation of the Standard Normal Integral
	// Gary R. Waissi and Donald F. Rossin

	double z = (x - this->mean) / this->stan_dev;
	double cdf = 1 / (1 + exp(-sqrt_pi*(-0.0004406*pow(z, 5) + 0.0418198*pow(z, 3) + 0.9*z)));
	return cdf;
}

void Probability_Node::reset() {
	this->probability_of_completion.clear();
	this->completion_time.clear();

	this->probability_of_completion.push_back(0.0);
	this->completion_time.push_back(0.0);

	this->probability_of_completion.push_back(1.0);
	this->completion_time.push_back(double(INFINITY));
}

void Probability_Node::print_out() {
	for (size_t i = 1; i < this->completion_time.size()-1; i++) {
		printf("    P( task[%i] = complete |%0.2f sec) = %0.2f \n", this->task_index, this->completion_time[i], this->probability_of_completion[i]);
	}
}
