#pragma once

#include <vector>
#include <iostream>

// This has never been tested!!!!

class Sorted_List
{
public:
	Sorted_List();
	~Sorted_List();

	bool get_val_by_key(const int &k, double &v, int &i);
	bool get_val_by_index(const int &i, double &v, int &k);
	void add_item(const int &k, const double &v);
	void remove_item_by_key(const int &k);
	void remove_item_by_index(const int &i);
	void print_list();

private:
	std::vector<int> keys;
	std::vector<double> vals;
};

