#include "Pose.h"
#include "Map_Node.h"
#include <math.h> 


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

double Pose::distance_to(Pose* &mn){
	return sqrt(pow(this->x - mn->get_x(),2) + pow(this->y - mn->get_y(),2));
}

double Pose::distance_to(Map_Node* &mn){
	return sqrt(pow(this->x - mn->get_x(),2) + pow(this->y - mn->get_y(),2));
}

Pose::~Pose(){}
