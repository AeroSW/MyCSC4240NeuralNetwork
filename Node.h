/*
 * Node.h
 *
 *  Created on: Sep 23, 2016
 *      Author: uge
 */

#ifndef NODE_H_
#define NODE_H_
#include<vector>

namespace AeroSW {
class Node {
	std::vector<double> weights;	// Stores the weights for inputs
	double value;		// Value to store base or sigmoidal value
	double delta;		// Delta to store the differential for expected value
	double bias;
	double sigmoid(double input);
	double summation(std::vector<double> inputs);
	unsigned int num_weights;
	public:
		Node(unsigned int num_weights);
		Node(std::vector<double> weights, unsigned int num_weights);
		~Node();
		bool set_value(double new_val);
		bool set_weights(std::vector<double> new_weights, unsigned int size);
		bool set_bias(double new_bias);
		double backward(double value);
		double forward(std::vector<double> inputs, unsigned int num_inputs);
		double get_delta();
		double get_value();
		double get_weight(unsigned int index);
		double get_bias();
		unsigned int get_num_weights();
};

} /* namespace AeroSW */

#endif /* NODE_H_ */
