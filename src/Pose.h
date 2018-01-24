#pragma once


class Pose{
public:
	Pose(const double &xi, const double &yi, const double &zi, const double &yawi);
	~Pose();

	double get_x() {return this->x; };
	double get_y() {return this->y; };
	double get_z() {return this->z; };
	double get_yaw() {return this->yaw; };

	void update_pose(const double &xi, const double &yi, const double &zi, const double wi);

private:
	double x,y,z,yaw;
};

