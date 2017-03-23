/*
 * NeuralNetDriver.cpp
 *
 *  Created on: Sep 26, 2016
 *      Author: uge
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstddef>
#include <cmath>
#include "Network.h"

using namespace std;

std::vector<std::string> split(const std::string &s, char delim){
	std::stringstream ss(s);
	std::string item;
	std::vector<std::string> tokens;
	while(std::getline(ss, item, delim)){
		tokens.push_back(item);
	}
	return tokens;
}

int main(int argc, char* argv[]){
	if(argc != 2)
	{
		std::cerr << "Incorrect number of arguments: <Program_Name> <Arg_1>\n";
		return 1;
	}
	/*
	 * 	FILE STUFF
	 */
	std::fstream file_stream(argv[1], std::fstream::in);
	// Read the information from argv[1]
	std::string data_file_name;
	file_stream >> data_file_name;
	std::string num_hid_lays_str; // Number of Hidden Layers as string
	file_stream >> num_hid_lays_str;
	std::string num_hid_nodes_str;
	file_stream >> num_hid_nodes_str; // Number of Hidden Nodes per Hidden Layer.
	std::string learning_rate_str;
	file_stream >> learning_rate_str;
	std::string err_tol_str;
	file_stream >> err_tol_str;
	file_stream.close(); // Close the file.
	// Open data_file_name
	file_stream.open(data_file_name, std::fstream::in);
	std::string scaling_factor_str;
	file_stream >> scaling_factor_str;
	std::string io_line;
	file_stream >> io_line;
	std::vector<std::string> examples_str;
	std::string temp;
	while(file_stream >> temp){
		if(temp[0] == 0) continue;
		examples_str.push_back(temp);
	}
	file_stream.close(); // Close the file.
	/*
	 * 	STRING MANIPULATION STUFF
	 */
	// How many hidden layers do we have as an int?
	unsigned int number_of_hidden_layers = stoi(num_hid_lays_str);
	unsigned int number_of_hidden_nodes = stoi(num_hid_nodes_str);
	unsigned int number_of_layers = number_of_hidden_layers + 2; // Add 2 for input output layers
	double learning_rate = stod(learning_rate_str);
	double duration = stod(err_tol_str);
	duration *= 60; // It is given in minutes, so convert it to seconds
	std::vector<std::string> parsed_str = split(io_line, ',');
	unsigned int num_input_nodes = stoul(parsed_str[0]);
	unsigned int num_output_nodes = stoul(parsed_str[1]);
	std::vector<unsigned int> blueprint(number_of_layers);
	blueprint[0] = num_input_nodes;
	blueprint[number_of_layers-1] = num_output_nodes;
	for(unsigned int i = 1; i < number_of_hidden_layers + 1; i++){
		blueprint[i] = number_of_hidden_nodes;
	}
	AeroSW::Network net(learning_rate, blueprint, number_of_layers);
	unsigned int num_examples = examples_str.size();
	double scaling_factor = stod(scaling_factor_str);
	std::vector<std::vector<double> > input_examples(num_examples);
	std::vector<std::vector<double> > output_examples(num_examples);
	for(unsigned int i = 0; i < num_examples; i++){
		parsed_str = split(examples_str[i], ',');
		unsigned int s_j = 0;
		std::vector<double> *inputs = new std::vector<double>(num_input_nodes);
		//std::vector<double> inputs(num_input_nodes);
		for(; s_j < num_input_nodes;s_j++){
			(*inputs)[s_j] = stod(parsed_str[s_j]);
		}
		input_examples[i] = *inputs;
		std::vector<double> *outputs = new std::vector<double>(num_output_nodes);
		for(unsigned int j = 0; j < num_output_nodes; j++){
			//std::cout << "Output: " << stod(parsed_str[s_j]) << '\n';
			(*outputs)[j] = stod(parsed_str[s_j++]);
		}
		output_examples[i] = *outputs;
	}
	std::cout << "Scaling Factor: " << scaling_factor << '\n';
	net.train(input_examples, output_examples, scaling_factor, num_examples, duration);
	//std::cout << "seg_fault in_feed\n";
	std::vector<std::vector<double> > result_set;
	for(unsigned int i = 0; i < num_examples; i++){
		 std::vector<double> results = net.feed(input_examples[i]);
		 result_set.push_back(results);
	}
	//std::cout << "nvm, seg_fault after feed\n";
	// Computing total error here
	double error_total = 0.0;
	for(unsigned int i = 0; i < num_examples; i++){
		std::vector<double> pre_determined_outputs = output_examples[i];
		std::vector<double> computed_outputs = result_set[i];
		for(unsigned int j = 0; j < num_output_nodes; j++){
			double error = abs(pre_determined_outputs[j] - (computed_outputs[j] * scaling_factor)) / pre_determined_outputs[j];
			error_total += error;
		}
	}
	// Printing Everything
	for(unsigned int i = 0; i < num_examples; i++){
		for(unsigned int j = 0; j < num_input_nodes; j++){
			std::cout << input_examples[i][j] << '\t';
		}
		for(unsigned int j = 0; j < num_output_nodes; j++){
			std::cout << output_examples[i][j] << '\t';
		}
		for(unsigned int j = 0; j < num_output_nodes; j++){
			std::cout << result_set[i][j] * scaling_factor;
			if(j != num_output_nodes-1)  std::cout << '\t';
		}
		std::cout << '\n';
	}
	std::cout << "TOTAL ERROR:\t" << error_total * 100 << "%\n";
	std::cout << "AVG ERROR:\t" << error_total * 100 / num_examples << "%\n";
	return 0;
}
