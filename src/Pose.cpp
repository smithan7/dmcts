#include "Pose.h"



Pose::Pose(const double &xi, const double &yi, const double &zi, const double &yawi){
	this->x = xi;
	this->y = yi;
	this->z = zi;
	this->yaw = yawi;
}


void Pose::update_pose(const double &xi, const double &yi, const double &zi, const double wi){
	this->x = xi;
	this->y = yi;
	this->z = zi;
	this->yaw = wi;
}

Pose::~Pose(){}
