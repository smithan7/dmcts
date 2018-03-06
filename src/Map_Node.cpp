#include "Map_Node.h"
#include "Agent.h"
#include "World.h"

#include <iostream>
#include <random>

Map_Node::Map_Node(const double &x, const double &y, const int &index, const double &p_active, const int &task_type, std::vector<double> &work, const cv::Scalar &color, const bool &flat_tasks, World* world){
	this->world = world;
	
	// where am I?
	this->x = x;
	this->y = y;
	this->loc = cv::Point2d(x, y);
	this->index = index;
	this->color = color;
	this->n_nbrs = 0;

	// set up the tasks
	this->n_reward_window_types = 4;
	this->task_type = task_type;
	this->agent_work = work;
	this->flat_tasks = flat_tasks;

	// range of rewards for tasks
	this->max_reward = world->get_max_task_reward(this->task_type);
	this->min_reward = world->get_min_task_reward(this->task_type);

	// range of time available for rewards and used to set rewards
	this->min_time = world->get_min_task_time(this->task_type);
	this->max_time = world->get_max_task_time(this->task_type);

	this->end_mission_time = world->get_end_time();


	// how much work does it take to complete this task
	this->min_work = world->get_min_task_work(this->task_type);
	this->max_work = world->get_max_task_work(this->task_type);

	// start setting it up
	this->active = false;
	if (p_active > world->rand_double_in_range(0.0, 1.0)) {
		// should I be active upon spawning?
		this->activate(world);
	}
}

void Map_Node::deactivate() {
	this->active = false;
	this->initial_reward = 0.0;
}


void Map_Node::update_task(World* world) {
	if (this->active) {
		if (world->get_c_time() > this->end_time || world->get_c_time() < this->start_time) {
			this->deactivate();
		}
	}
}

double Map_Node::get_acted_upon(Agent* agent) {
	// this agent says it is working on me, check

	// are they actually at my location
	if (agent->at_node(this->index)) {
		// they are actually at my location, what type are they?
		int agent_type = agent->get_type();
		double agent_work = this->agent_work[agent_type];
		this->remaining_work -= agent_work;
		if (this->remaining_work <= 0.0) {
			agent->set_collected_reward(this->get_reward_at_time(agent->get_world()->get_c_time()));
			this->deactivate();
			return agent_work + this->remaining_work;
		}
		else {
			return agent_work;
		}
	}
	else {
		return 0.0;
	}
}

double Map_Node::get_time_to_complete(Agent* agent, World* world) {
	double agent_work = this->agent_work[agent->get_type()];
	return this->remaining_work / agent_work;
}

void Map_Node::activate(World* world) {
	this->active = true;
	this->start_time = world->get_c_time();

	// what type of reward window will I have?
	//std::cout << "Map_Node::activate: fixed typing" << std::endl;
	this->reward_window_type = rand() % this->n_reward_window_types;
	// what is my initial reward?
	//std::cout << "Map_Node::activate: fixed reward value" << std::endl;
	this->initial_reward = (this->max_reward - this->min_reward) * double(rand()) / double(RAND_MAX) + this->min_reward;
	// how much work to complete me?
	this->remaining_work = (this->max_work - this->min_work) * double(rand()) / double(RAND_MAX) + this->min_work;

	if (this->reward_window_type == 0) {
		// constant value, do nothing
		this->end_time = double(INFINITY);
	}
	else if (this->reward_window_type == 1) {
		// linear decline
		double active_interval = (this->max_time - this->min_time) * double(rand()) / double(RAND_MAX) + this->min_time;
		this->end_time = this->start_time + active_interval;
		this->reward_slope = (this->min_reward - this->initial_reward) / active_interval;
		this->reward_offset = this->initial_reward - this->reward_slope*this->start_time;
	}
	else if (this->reward_window_type == 2) {
		// constant window
		double active_interval = (this->max_time - this->min_time) * double(rand()) / double(RAND_MAX) + this->min_time;
		this->end_time = this->start_time + active_interval;
	}
	else if (this->reward_window_type == 3) {
		// exponential decline
		double active_interval = (this->max_time - this->min_time) * double(rand()) / double(RAND_MAX) + this->min_time;
		this->end_time = this->start_time + active_interval;
		this->reward_decay = log(this->min_reward / this->initial_reward) / active_interval;
	}
}

double Map_Node::get_reward_at_time(double time) {

	if (!this->active || time > this->end_time || time < this->start_time || time > this->end_mission_time) {
		//std::cerr << "this->acitve: " << this->active << std::endl;
		//std::cerr << "time > this->end_time: " << time << " > " << this->end_time << std::endl;
		//std::cerr << "time < this->start_time: " << time << " < " << this->start_time << std::endl;
		//std::cerr << "time > this->end_mission_time: " << time << " > " << this->end_mission_time << std::endl;
		return 0.0;
	}

	if (this->flat_tasks){ // all tasks have equal rewards, sooner you get to them the better!
		return std::max(150.0 - time,0.0);
	}

	if (this->reward_window_type == 0) {
		// constant value
		return this->initial_reward;
	}
	else if (this->reward_window_type == 1) {
		// linear decline
		return this->reward_slope*time + reward_offset;
	}
	else if (this->reward_window_type == 2) {
		// constant window, time constraints ensure that I am good
		return this->initial_reward;
	}
	else if (this->reward_window_type == 3) {
		// exponential decline
		double dt = time - this->start_time;
		double r_out = this->initial_reward * exp(this->reward_decay * dt);
		return r_out;
	}
	else {
		return 0.0;
	}
}

Map_Node::~Map_Node(){}

void Map_Node::add_nbr(const int &nbr, const double &dist, const double &obs_cost) {
	this->nbrs.push_back(nbr);
	this->nbr_distances.push_back(dist);
	this->n_nbrs++;
	this->nbr_obstacle_costs.push_back(obs_cost);
}

bool Map_Node::get_nbr_travel_cost(const int &index, const bool &pay_obstacle_cost, double &nbr_cost) {
	if (index < this->n_nbrs) {
		if (pay_obstacle_cost) {
			nbr_cost = this->nbr_obstacle_costs[index];
		}
		else {
			nbr_cost = this->nbr_distances[index];
		}
		return true;
	}
	return false;
}

bool Map_Node::get_nbr_obstacle_cost(const int &index, double &nbr_cost) {
	if (index < this->n_nbrs) {
		nbr_cost = this->nbr_obstacle_costs[index];
		return true;
	}
	return false;
}

bool Map_Node::get_nbr_distance(const int &index, double &nbr_dist) {
	if (index < this->n_nbrs) {
		nbr_dist = this->nbr_distances[index];
		return true;
	}
	return false;
}

bool Map_Node::get_nbr_i(const int &index, int &nbr_index) {
	if (index < this->n_nbrs) {
		nbr_index = this->nbrs[index];
		return true;
	}
	return false;
}

bool Map_Node::is_nbr(const int &n) {
	for (int i = 0; i < this->n_nbrs; i++) {
		if (this->nbrs[i] == n) {
			return true;
		}
	}
	return false;
}
