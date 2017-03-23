/*
 * Network.h
 *
 *  Created on: Sep 23, 2016
 *      Author: uge
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "Node.h"
#include <vector>
#include <memory>
#include <stdint.h>

namespace AeroSW {

class Network {
	// Vector holding pointers for nodes in network
	std::vector<std::shared_ptr<Node> > nodes;
	// Total number of nodes in this network.  Summation(Nodes_per_Layer[i])
	// for each index i in Nodes_per_Layer
	unsigned int num_nodes;
	// Indicates number of layers in this network, also, it represents length
	// of nodes_per_layer array.
	unsigned int num_layers;
	// This tracks how many nodes are in each layer of the network where an
	// index + 1 represents the layer and the integer stored at that index is
	// the number of nodes.
	std::vector<uint32_t> nodes_per_layer;
	// alpha is the learning rate for our Neural Network.
	double alpha;

	public:
		/*
		 * Constructor
		 * Description:	This constructor builds the neural network based on blueprints passed in by the user.
		 * Parameters:
		 * 		a:			alpha or the learning rate for this network
		 * 		blueprint:	integer array storing the number of nodes in layer l, where l is an index of the array - 1.
		 * 		nl:			number of layers in the network, or the length of blueprint
		 * returns:
		 * 		A new UNTRAINED neural network
		 */
		Network(double a, std::vector<uint32_t> blueprint, unsigned int nl);
		/*
		 * Destructor
		 * Description:	Deallocates the memory for each node in std::vector<Node*> and unsigned int* nodes_per_layer
		 */
		~Network();
		/*
		 * Method:		train_tim - Train for Time Period
		 * Description:	This method trains the designed neural network for x amount of time.
		 * Parameters:
		 * 		vector<vector<double>> ex_in:	a vector of example inputs
		 * 		vector<vector<double>> ex_ou:	a vector of example outputs
		 * 		double scaling_factor:			a value to scale outputs with for accurate training results
		 * 		unsigned int num_examples:		an unsigned integer representing the number of elements in the vector.
		 * 		unsigned int duration:			an unsigned long representing the number of seconds this program trains
		 * 										the neural network.
		 * 	Returns:
		 * 		Nothing
		 */
		void train(std::vector<std::vector<double> > ex_in, std::vector<std::vector<double> > ex_ou, double scaling_factor, unsigned int num_examples, unsigned long duration);
		/*
		 * Method:		feed
		 * Description:	The method takes a set of inputs and feeds them into the neural network for decisions
		 * Parameters:
		 * 		vector<double> inputs:		A vector of inputs to be fed into the Neural Network.
		 * Returns:
		 * 		Returns a pointer for a vector<double> which holds the results.
		 */
		std::vector<double> feed(std::vector<double> inputs);
};

} /* namespace AeroSW */

#endif /* NETWORK_H_ */
