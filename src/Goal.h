#pragma once

#include <vector>

class Goal
{
public:
	Goal();
	~Goal();

	int get_index() { return this->index; };
	
	double get_current_reward() { return this->current_reward; };
	double get_arrival_reward() { return this->arrival_reward; };
	double get_completion_reward() { return this->completion_reward; };
	
	double get_distance() { return this->distance; };
	
	double get_current_time() { return this->current_time; };
	double get_arrival_time() { return this->arrival_time; };
	double get_completion_time() { return this->completion_time; };

	double get_completion_value() { return this->completion_value; };
	std::vector<int> get_path() { return this->path; };

	void set_index(int i) { this->index = i; };
	
	void set_current_reward(double i) { this->current_reward = i; };
	void set_arrival_reward(double i) { this->arrival_reward = i; };
	void set_completion_reward(double i) { this->completion_reward = i; };
	
	void set_distance(double i) { this->distance = i; };
	
	void set_current_time(double i) { this->current_time = i; };
	void set_arrival_time(double i) { this->arrival_time = i; };
	void set_completion_time(double i) { this->completion_time = i; };

	void set_completion_value(double i) { this->completion_value = i; };
	void set_path(const std::vector<int> &path) { this->path = path; };

private:
	int index;
	double current_reward, arrival_reward, completion_reward;
	double distance;
	double current_time, arrival_time, completion_time;
	double completion_value;
	std::vector<int> path;
};

