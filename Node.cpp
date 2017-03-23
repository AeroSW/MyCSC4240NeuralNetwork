/*
 * Node.cpp
 *
 *  Created on: Sep 23, 2016
 *      Author: uge
 */

#include "Node.h"
#include <math.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
namespace AeroSW {
	//
	//	PUBLIC FUNCTIONS
	//
	Node::Node(unsigned int nw) {
		num_weights = nw;
		for(unsigned int i = 0; i < num_weights; i++){
			weights.push_back(1.0);
		}
		delta = 0.0;
		value = 1.0;
		bias = 0.0;
	}
	Node::Node(std::vector<double> ws, unsigned int nw){
		num_weights = nw;
		for(unsigned int i = 0; i < nw; i++){
			weights.push_back(ws[i]);
		}
		delta = 0.0;
		value = 1.0;
		bias = 0.0;
	}
	Node::~Node()
	{
		weights.clear();
	}
	bool Node::set_bias(double new_bias)
	{
		bias = new_bias;
		return true;
	}
	bool Node::set_value(double new_val)
	{
		value = new_val;
		return true;
	}
	bool Node::set_weights(std::vector<double> new_weights, unsigned int size)
	{
		if(size != num_weights) return false;
		for(unsigned int i = 0; i < size; i++)
			weights[i] = new_weights[i];
		return true;
	}
	double Node::backward(double input)
	{
		double dif_sig = value * (1 - value);
		delta = dif_sig * input;
		return delta;
	}
	double Node::forward(std::vector<double> inputs, unsigned int num_inputs)
	{
		if(num_inputs != num_weights)
		{
			std::cerr << "Improper inputs\n";
			std::exit(-1);
		}
		double summed = summation(inputs);
		summed += bias;
		double result = sigmoid(summed);
		value = result;
		return value;
	}
	double Node::get_delta()
	{
		return delta;
	}
	double Node::get_value()
	{
		return value;
	}
	double Node::get_weight(unsigned int index)
	{
		return weights[index];
	}
	double Node::get_bias()
	{
		return bias;
	}
	unsigned int Node::get_num_weights()
	{
		return num_weights;
	}
	//
	//	PRIVATE FUNCTIONS
	//
	double Node::sigmoid(double input)
	{
		double exponential = exp(-input);
		double result = 1 / (1 + exponential);
		return result;
	}
	double Node::summation(std::vector<double> inputs)
	{
		std::vector<double> temp_array(num_weights);
		for(unsigned int i = 0; i < num_weights; i++)
		{
			temp_array[i] = weights[i] * inputs[i];
		}
		double summ = 0.0;
		for(unsigned int i = 0; i < num_weights; i++)
			summ += temp_array[i];
		return summ;
	}
}

