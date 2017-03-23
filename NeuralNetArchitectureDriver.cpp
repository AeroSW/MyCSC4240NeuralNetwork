// NeuralNetworkArchitectureDriver.cpp

/*
 *	Observations During K-Fold Validation
 *	
 *	During the execution of the K-Fold Cross Validation,
 *	you wanted us to make an observation of what was occurring
 *	as we increased the number of nodes and hidden layers for
 *	the architecture trials.  I noticed my architectures would
 *	gradually lean toward one layer typically throughout the tests
 *	unless the randomization was such that a more complex
 *	architecture's error was low enough to make any simpler one
 *	inadequate. Typically, once I made it above six nodes per layer
 *	the program would then cycle between a 3,5,1 architecture and a
 *	3,3,1 architecture as being the most optimal for each run.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <cstddef>
#include <cmath>
#include "Validation.h"
#include "Network.h"
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
	std::fstream file_stream(argv[1], std::fstream::in);
	if(file_stream.is_open()){
		std::cout << "File opened\n";
	}
	else{
		std::cerr << "File did not open \n";
		std::exit(1);
	}
	// Parse the file
	std::string data_file_name;
	file_stream >> data_file_name;
	std::string num_bins_str; // Number of bins to sort data points into
	file_stream >> num_bins_str;
	std::string max_hid_layers_str;
	file_stream >> max_hid_layers_str; // Max Hidden Layers for Network
	std::string max_hid_nodes_str;
	file_stream >> max_hid_nodes_str; // Max number of nodes per hidden layer
	std::string learning_rate_str; // Learning rate for the networks
	file_stream >> learning_rate_str;
	std::string err_tol_str; // Time in minutes to train each network.
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
		//std::cout << temp << std::endl;
		examples_str.push_back(temp);
	}
	file_stream.close(); // Close the file.
	
	/*
	 * 	STRING MANIPULATION STUFF
	 */
	// How many hidden layers do we have as an int?
	unsigned int num_folds = std::stoi(num_bins_str);
	unsigned int max_hidden_layers = std::stoi(max_hid_layers_str);
	unsigned int max_hidden_nodes = std::stoi(max_hid_nodes_str);
	double learning_rate = std::stod(learning_rate_str);
	double duration = std::stod(err_tol_str);
	duration *= 60; // Convert from minutes to seconds.
	std::vector<std::string> parsed_str = split(io_line, ',');
	unsigned int num_input_nodes = stoul(parsed_str[0]);
	unsigned int num_output_nodes = stoul(parsed_str[1]);
	double scaling_factor = stod(scaling_factor_str);
	
	// Parse the string variants of the data points.
	std::vector<double*> data_list;
	for(unsigned int data_counter = 0; data_counter < examples_str.size(); data_counter++){
		//std::cout << "Current Linked_List Size: " << data_list.size() << '\n';
		std::vector<std::string> p_str = split(examples_str[data_counter],','); // Split the string
		double* data_array = new double[p_str.size()];
		for(unsigned int i = 0; i < p_str.size(); i++){
			double mediate = stod(p_str[i]);
			//std::cout << mediate << ',';
			data_array[i] = mediate;
		}
		//std::cout << std::endl;
		data_list.push_back(data_array);
	}
	
	uint32_t esti_time = 0;
	uint32_t num_bps = 0;
	for(uint32_t i = 0; i <= max_hidden_layers; i++){
		num_bps += (pow(max_hidden_nodes, i));
	}
	esti_time = num_bps * num_folds * duration;
	uint32_t num_threads = std::thread::hardware_concurrency();
	if(num_threads == 0) num_threads = 1;
	esti_time = ceil(esti_time / num_threads);
	
	
	time_t curr_time = time(0);   // get time now
    struct tm* now = localtime(&curr_time);
	
	
	std::cout << "Estimated Time for Completion:\t" << esti_time << " seconds\n";
	std::cout << "Time may excede estimate\n\n";
	std::cout << "Starting Validation\t" << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec << "\n";
	std::cout << "Improvement\tBlueprint\tError * Number_of_Nodes\tError\n";
	optarc_t best = k_fold_cross(num_input_nodes, num_output_nodes, max_hidden_nodes, max_hidden_layers, num_folds, learning_rate, scaling_factor, duration, data_list);
	
	std::cout << "\n";
	std::cout << "Best Architecture\n";
	std::cout << "=================\n\n";
	std::cout << "BP:\t[";
	for(uint32_t i = 0; i < best.nl; i++){
		std::cout << best.bp[i];
		if(i + 1 != best.nl) std::cout << ',';
	}
	std::cout << "]\n";
	std::cout << "Error:\t" << best.err << '\n';
	return 0;
}
