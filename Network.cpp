/*
 * Network.cpp
 *
 *  Created on: Sep 23, 2016
 *      Author: uge
 */

#include "Network.h"
#include <iostream>
#include <random>
#include <string>
#include <time.h>
#include <omp.h>
namespace AeroSW {

	Network::Network(double a, std::vector<uint32_t> blueprint, unsigned int nl)
	{
		alpha = a;
		num_layers = nl;
		unsigned int summ = 0;
		for(unsigned int i = 0; i < nl; i++) summ += blueprint[i];
		num_nodes = summ;
		for(unsigned int i = 0; i < num_layers; i++)
			nodes_per_layer.push_back(blueprint[i]);
		for(unsigned int l = 0; l < num_layers; l++){
			unsigned int prev_num;
			if(l != 0)
				prev_num = nodes_per_layer[l-1];
			else
				prev_num = 0;
			unsigned int num_reg_nodes = 0;
			num_reg_nodes = nodes_per_layer[l];
			unsigned int s_i = 0;
			for(unsigned int i = 0; i < l; i++) s_i += nodes_per_layer[i];
			unsigned int e_i = s_i + num_reg_nodes;
			for(unsigned int i = s_i; i < e_i; i++){
				std::shared_ptr<Node> ptr_ref(new Node(prev_num));
				nodes.push_back(ptr_ref);
			}
		}
	}

	Network::~Network() {
		nodes.clear();
		nodes_per_layer.clear();
	}
	void Network::train(std::vector<std::vector<double> > ex_in, std::vector<std::vector<double> > ex_ou, double scaling_factor, unsigned int num_examples, unsigned long duration)
	{
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_real_distribution<double> dist(-0.1, 0.1);
		for(unsigned int index = 0; index < num_nodes; index++){
			unsigned int nw = nodes[index]->get_num_weights();
			std::vector<double> new_weights(nw); // Create an array to hold new weights.
			for(unsigned int w_index=0;w_index < nw; w_index++){
				double value = dist(gen);
				new_weights[w_index] = value; // Randomize new weights
			}
			nodes[index]->set_weights(new_weights, nw); // Change the node's weights.
			nodes[index]->set_bias(dist(gen));
		}
		// We need to iterate until the stopping criteria is met
		// Declare the stopping criteria
		unsigned long init = time(0);
		unsigned long curr = time(0);
		// Loop/Iterate now
		while(curr - init < duration){
			// Iterate through list of examples
			for(unsigned int ex_i = 0; ex_i < num_examples; ex_i++){
				// Set the input nodes to be the example inputs
				if(true){
					// Separating code
					unsigned int s_i = 0;
					unsigned int e_i = s_i + nodes_per_layer[0];
					for(unsigned int i = s_i; i < e_i; i++){
						nodes[i]->set_value(ex_in[ex_i][i]);
					}
				}
				for(unsigned int l = 1; l < num_layers; l++){
					std::vector<double> inputs(nodes_per_layer[l-1]);
					// Grab values from previous layer to use as inputs.
					{ // Splitting up work here
						// Need to compute starting index
						unsigned int s_i = 0;
						for(unsigned int i = 0; i < l-1; i++) s_i += nodes_per_layer[i];
						unsigned int e_i = nodes_per_layer[l-1] + s_i;
						for(unsigned int i = s_i; i < e_i; i++){
							inputs[i-s_i] = nodes[i]->get_value();
						}
					}
					// Feed the inputs into the current layer
					{
						unsigned int s_i = 0;
						for(unsigned int i = 0; i < l; i++) s_i += nodes_per_layer[i];
						unsigned int e_i = nodes_per_layer[l] + s_i;
						for(unsigned int i = s_i; i < e_i; i++){
							nodes[i]->forward(inputs, nodes_per_layer[l-1]);
						}
					}
					//delete inputs;
				}
				// Propagate Back
				{
					unsigned int s_i = 0;
					for(unsigned int i = 0; i < num_layers-1; i++) s_i += nodes_per_layer[i];
					unsigned int e_i = s_i + nodes_per_layer[num_layers - 1];
					for(unsigned int i = s_i; i < e_i; i++){
						double temp = (ex_ou[ex_i][i-s_i] / scaling_factor) - nodes[i]->get_value();
						nodes[i]->backward(temp);
					}
				}
				for(int l = num_layers - 2; l >= 0; l--){
					unsigned int s_i = 0;
					for(unsigned int i = 0; i < (unsigned int)l; i++) s_i += nodes_per_layer[i];
					unsigned int e_i = s_i + nodes_per_layer[l];
					for(unsigned int i = s_i; i < e_i; i++){
						unsigned int s_j = 0;
						for(unsigned int j = 0; j < (unsigned int)l+1; j++) s_j += nodes_per_layer[j];
						unsigned int e_j = s_j + nodes_per_layer[l+1];
						double summ = 0;
						for(unsigned int j = s_j; j < e_j; j++){
							summ += (nodes[j]->get_weight(i - s_i) * nodes[j]->get_delta());
						}
						nodes[i]->backward(summ);
					}
				}
				//#pragma omp critical
				{
					for(unsigned int l = 1; l < num_layers; l++){
						unsigned int s_j = 0;
						for(unsigned int j = 0; j < l; j++) s_j += nodes_per_layer[j];
						unsigned int e_j = s_j + nodes_per_layer[l];
						for(unsigned int j = s_j; j < e_j; j++){
							unsigned int nw = nodes[j]->get_num_weights();
							std::vector<double> new_weights(nw);
							unsigned int s_i = 0;
							for(unsigned int i = 0; i < l - 1; i++) s_i += nodes_per_layer[i];
							unsigned int e_i = s_i + nodes_per_layer[l-1];
							for(unsigned int i = s_i; i < e_i; i++){
								unsigned int q = i - s_i;
								//std::cout << "L:\t" << l << "\tJ:\t" << j << "\tI:\t" << i << "\tQ:\t" << q << '\n';
								new_weights[i - s_i] = (nodes[j]->get_weight(q) + (alpha * nodes[i]->get_value() * nodes[j]->get_delta()));
							}
							nodes[j]->set_weights(new_weights, nw);
							double b = nodes[j]->get_bias();
							double nb = b + (alpha * nodes[j]->get_delta());
							nodes[j]->set_bias(nb);
						}
					}
				}
			}
			curr = time(0);
		}
	}
	std::vector<double> Network::feed(std::vector<double> inputs){
		if(true){
			// Separating code
			unsigned int s_i = 0;
			unsigned int e_i = s_i + nodes_per_layer[0];
			for(unsigned int i = s_i; i < e_i; i++){
				nodes[i]->set_value(inputs[i]);
			}
		}
		for(unsigned int l = 1; l < num_layers; l++){
			std::vector<double> inputs(nodes_per_layer[l-1]);
			// Grab values from previous layer to use as inputs.
			if(true){ // Splitting up work here
				// Need to compute starting index
				unsigned int s_i = 0;
				for(unsigned int i = 0; i < l-1; i++) s_i += nodes_per_layer[i];
				unsigned int e_i = nodes_per_layer[l-1] + s_i;
				for(unsigned int i = s_i; i < e_i; i++){
					inputs[i-s_i] = nodes[i]->get_value();
				}
			}
			// Feed the inputs into the current layer
			if(true){
				unsigned int s_i = 0;
				for(unsigned int i = 0; i < l; i++) s_i += nodes_per_layer[i];
				unsigned int e_i = nodes_per_layer[l] + s_i;
				for(unsigned int i = s_i; i < e_i; i++){
					nodes[i]->forward(inputs, nodes_per_layer[l-1]);
				}
			}
			//delete inputs;
		}
		std::vector<double> outputs;
		if(true){
			unsigned int s_j = 0;
			for(unsigned int j = 0; j < num_layers-1; j++) s_j += nodes_per_layer[j];
			unsigned int e_j = s_j + nodes_per_layer[num_layers-1];
			for(unsigned int j = s_j; j < e_j; j++){
				outputs.push_back(nodes[j]->get_value());
			}
		}
		return outputs;
	}
} /* namespace AeroSW */
