#include "Sorted_List.h"

// This has never been tested!!!!!

Sorted_List::Sorted_List(){
	while (true) {
		std::cout << "Sorted List Created but never tested!!!!" << std::endl;
	}
}


Sorted_List::~Sorted_List(){}


bool Sorted_List::get_val_by_key(const int &k, double &v, int &i) {
	for (size_t j = 0; j < this->keys.size(); j++) {
		if (this->keys[j] == k) {
			i = int(j);
			v = this->vals[j];
			return true;
		}
	}
	return false;
}

bool Sorted_List::get_val_by_index(const int &i, double &v, int &k) {
	if (i > 0 && i < this->vals.size()) {
		k = this->keys[i];
		v = this->vals[i];
		return true;
	}
	return false;
}

void Sorted_List::add_item(const int &k, const double &v) {

	for (size_t i = 0; i < this->vals.size(); i++) {
		if (v < this->vals[i]) {
			this->vals.insert(this->vals.begin() + i, v);
			this->keys.insert(this->keys.begin() + i, k);
			return;
		}
	}

	this->vals.push_back(v);
	this->keys.push_back(k);
}

void Sorted_List::remove_item_by_key(const int &k) {
	for (size_t j = 0; j < this->keys.size(); j++) {
		if (this->keys[j] == k) {
			this->keys.erase(this->keys.begin() + j);
			this->vals.erase(this->vals.begin() + j);
			return;
		}
	}
}

void Sorted_List::remove_item_by_index(const int &i) {
	this->keys.erase(this->keys.begin() + i);
	this->vals.erase(this->vals.begin() + i);
}

void Sorted_List::print_list() {
	std::printf("Sorted List has %i items", int(this->keys.size()));
	for (size_t i = 0; i < this->keys.size(); i++) {
		std::printf("     [%i]: %i / 0.4%f", int(i), this->keys[i], this->vals[i]);
	}
}