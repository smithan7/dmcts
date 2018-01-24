#define _CRT_SECURE_NO_DEPRECATE

#include "World.h"

#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

	// builds the complete world, maps, node, agents, tasks
	int params = -1; // 5 tasks with 1 agent 5673, 100 tasks with 0.25 p_active and 1 agent 15591, 14310 20 nodes, 1 agent used to debug planning
	bool display_map = true;
	bool score_run = false;

	std::vector<cv::String> task_selection_methods;

	//task_selection_methods.push_back("impact_completion_value");
	//task_selection_methods.push_back("impact_completion_reward");
	//task_selection_methods.push_back("value_completion");
	task_selection_methods.push_back("greedy_completion_reward");
	//task_selection_methods.push_back("greedy_completion_time");
	//task_selection_methods.push_back("greedy_current_reward");
	//task_selection_methods.push_back("random_task");
	//task_selection_methods.push_back("mcts_task_by_completion_reward");
	//task_selection_methods.push_back("mcts_task_by_completion_value");
	//task_selection_methods.push_back("mcts_task_by_completion_reward_impact");
	//task_selection_methods.push_back("mcts_task_by_completion_reward_impact_before_and_after");
	task_selection_methods.push_back("mcts_task_by_completion_reward_impact_optimal"); // this has p = 1.0
	//task_selection_methods.push_back("mcts_task_by_completion_reward_impact_fixed"); // this has p = 0.5
	// not complete task_selection_methods.push_back("mcts_task_by_completion_value_impact");

	int sim_iters = 50;
	for (int i = 0; i < sim_iters; i++) {
		std::cout << "Test iteration " << i << " of " << sim_iters << std::endl;
		World world = World(params, display_map, score_run, task_selection_methods);
	}
	return 1;
}
