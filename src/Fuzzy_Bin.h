#pragma once
#include <vector>

class Fuzzy_Bin
{
public:
	Fuzzy_Bin();
	~Fuzzy_Bin();
	void init(std::vector<float> x, std::vector<float> y);

	float get_weight(float input);
	std::vector<float> x;
	std::vector<float> y;
};

