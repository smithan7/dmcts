#include "Fuzzy_Bin.h"

Fuzzy_Bin::Fuzzy_Bin() {}

void Fuzzy_Bin::init(std::vector<float> x_in, std::vector<float> y_in) {
	this->x = x_in;
	this->y = y_in;
}

Fuzzy_Bin::~Fuzzy_Bin() {}

float Fuzzy_Bin::get_weight(float input) {
	if (input < this->x[0] || input > this->x[2]) {
		return 0.0f;
	}

	if (input > this->x[0] && input < this->x[1]) {
		float m = (this->y[1] - this->y[0]) / (this->x[1] - this->x[0]);
		float b = this->y[1] - m*this->x[1];
		float w = m*input + b;
		return w;
	}

	if (input > this->x[1] && input < this->x[2]) {
		float m = (this->y[2] - this->y[1]) / (this->x[2] - this->x[1]);
		float b = this->y[2] - m*this->x[2];
		float w = m*input + b;
		return w;
	}

	if (input == this->x[1]) {
		return this->y[1];
	}

	return 0.0f;
}