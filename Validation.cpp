#include <omp.h>
#include <math.h>
#include <random>
#include <thread>
#include <iostream>
#include "Validation.h"
/*
 *	Function Name:	monte_carlo
 *	Description:	The function finds the best valid
 *					architecture, based on the input
 *					limitations, and returns it to the
 *					user.
 *					
 *	Parameters:
 *		uint32_t ni				Total number of inputs for the
 *								networks.
 *								
 *		uint32_t no				Total number of outputs for the
 *								networks.
 *								
 *		uint32_t ln				The highest number of possible
 *								nodes per hidden layer.
 *								
 *		uint32_t ll				The highest number of hidden la-
 *								yers allowed in a network.
 *								
 *		uint32_t num_bins		The number of bins to divide
 *								the training data into.
 *								
 *		double lr				The learning rate for the networks.
 *		
 *		double sf				The scaling factor for outputs in
 *								the networks.
 *								
 *		double dur 				The duration to train each network
 *								for.
 *								
 *		vector<double*> data	The data inputs and outputs for	|
 *								training.
 *								
 *	Returns:
 *		This function returns the network, its blueprint, its number
 *		of layers, and its error in a structure of type optarc_t.
 */
optarc_t k_fold_cross(uint32_t ni, uint32_t no, uint32_t ln, uint32_t ll,
					 uint32_t num_bins, double lr, double sf, double dur,
					 std::vector<double*> data){
	// Initialize the random devices for random number generation.
	std::random_device rd_bin;
	std::random_device rd_ind;
	std::mt19937_64 gen_bin(rd_bin());
	std::mt19937_64 gen_ind(rd_ind());
	std::uniform_int_distribution<int> dist_bin(0, num_bins-1); // 0-based
	std::uniform_int_distribution<int> dist_ind(0, (data.size() - 1)); // 0-based
	// Random Number Generators Initialized!
	// Initialize the container for the bins.
	std::vector<std::vector<std::vector<double> > > bins;
	// bins
	/*
	 *	Outer vector holds the bins
	 *	Middle vector simulates the bins and holds vectors containing the data
	 *	Inner vector holds input and output data
	 */
	// Create the individual bins.
	for(unsigned int bin_counter = 0; bin_counter < num_bins; bin_counter++){
		std::vector<std::vector<double> > t_vect;
		bins.push_back(t_vect);
	}
	//unsigned int quantity_of_data = data.size(); // Get the quantity of information we are dividing.
	//unsigned int quant_per_bin = quantity_of_data / num_bins; // Find the amount of information each bin will store.
	// Split the data
	unsigned int bin_counter = 0; // Initialize a counter for bin.
	std::vector<unsigned int> visited; // visited list.
	// Loop until number of visited items is equivilent to the quantity of data we have.
	while(visited.size() < data.size()){
		unsigned int ind = (unsigned) dist_ind(gen_ind);
		// Iterate over the visited list
		bool v_flag = false;
		for(unsigned int visit_counter = 0; visit_counter < visited.size(); visit_counter++){
			if(visited[visit_counter] == ind){ // Is this index in the visited list
				v_flag = true; // Get new random index if visited before
				break;
			}
		}
		if(v_flag){
			continue;
		}
		visited.push_back(ind);
		// Create references for the inputs and the outputs.
		std::vector<double> data_ref;
		for(unsigned int data_counter = 0; data_counter < (ni + no); data_counter++){
			data_ref.push_back(data[ind][data_counter]);
		}
		// Push the vector holding the data values into the appropriate bin.
		bins[bin_counter].push_back(data_ref);
		// Increment bin_counter, if it is equal to num_bins, reinitialize it to 0.
		bin_counter++;
		bin_counter %= num_bins;
	}
	// Bins are now constructed with input data inside of them.
	// We should now create an object to store our best architecture thus far within.
	optarc_t best;
	// Time to create the initial blueprint. To make it simple,
	// it will just be the input and output layers.
	best.bp = std::vector<uint32_t>();
	best.nl = 0;
	best.err = 999999999.999;
	best.nn = 1;
	// Initialize the network's structure.
	best.net_ref = 0;
	// Find how many processors are on this machine.
	unsigned int num_threads = std::thread::hardware_concurrency(); // Find # of cores
	if(num_threads == 0) // Assume 1 core for systems w/out multiple cores
		num_threads = 1;
	// We have an error now to compare other results with.
	// We need to construct the other blueprints now.
	std::vector<std::vector<unsigned int> > bps = construct_blueprints(ln, ll, ni, no);
	// Blueprints are now constructed.
	uint32_t num_slices = bps.size() / num_threads;
	std::vector<shared_ptr<Network> > networks;
	for(uint32_t net_id; net_id < bps.size(); net_id++){
		uint32_t bp_size = bps[net_id].size();
		shared_ptr<Network> temp_ref(new Network(lr, bps[net_id], bp_size));
		networks.push_back(temp_ref);
	}
	#pragma omp parallel num_threads(num_threads) firstprivate(networks, dist_bin, gen_bin, bps, bins, dur, sf, lr, no, ni)
	{
		uint32_t my_rank = omp_get_thread_num();
		uint32_t my_si = my_rank * num_slices;
		uint32_t my_ei;
		if(my_rank == num_threads-1)
			my_ei = networks.size();
		else
			my_ei = my_si + num_slices;
		// Here we need to start the outer for loop, to iterate through every blueprint in bps.
		for(uint32_t i = my_si; i < my_ei; i++){
			double curr_err = 0.0;
			for(uint32_t bin_counter = 0; bin_counter < num_bins; bin_counter++){
				std::vector<std::vector<double> > inputs;
				std::vector<std::vector<double> > outputs;
				for(uint32_t bc = 0 /*bc bincounter*/; bc < num_bins; bc++){
					if(bc == bin_counter) continue;
					for(uint32_t dc = 0; dc < bins[bc].size(); dc++){
						std::vector<double> vect_in(ni);
						std::vector<double> vect_out(no);
						for(uint32_t ic = 0; ic < ni; ic++){
							vect_in[ic] = bins[bc][dc][ic];
						}
						for(uint32_t oc = 0; oc < no; oc++){
							vect_out[oc] = bins[bc][dc][oc+ni];
						}
						inputs.push_back(vect_in);
						outputs.push_back(vect_out);
						//delete vect_in;
						//delete vect_out;
					}
				}
				networks[i]->train(inputs, outputs, sf, inputs.size(), dur);
				curr_err += test_network(networks[i], bins[bin_counter], ni, no, sf);
			}
			curr_err = (curr_err / num_bins) * 100;
			#pragma omp critical
			{
				uint32_t num_nodes = 0;
				for(uint32_t j = 0; j < bps[i].size(); j++){
					num_nodes += bps[i][j];
				}
				if((curr_err*num_nodes) < (best.err*best.nn)){
					std::cout << "Success\t\t";
					std::cout<< "[";
					for(uint32_t j = 0; j < bps[i].size();j++){
						std::cout << bps[i][j];
						if(j+1 != bps[i].size()) std::cout << ',';
					}
					if(bps[i].size() > 3)
						std::cout << "]\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					else
						std::cout << "]\t\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					if(curr_err < best.err)
						std::cout << curr_err << " < " << best.err << std::endl;
					else
						std::cout << curr_err << " > " << best.err << std::endl;
					//delete best.net_ref; // Free up pre-occupied memory.
					best.net_ref = networks[i]; // Store the reference to this network
					//delete best.bp; // Free up pre-occupied memory
					best.bp = bps[i];
					best.nl = bps[i].size();
					best.nn = num_nodes;
					best.err = curr_err; // Store this error.
				}
				else
				{
					std::cout << "Failure\t\t";
					std::cout<< "[";
					for(uint32_t j = 0; j < bps[i].size();j++){
						std::cout << bps[i][j];
						if(j+1 != bps[i].size()) std::cout << ',';
					}
					if(bps[i].size() > 3)
						std::cout << "]\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					else
						std::cout << "]\t\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					if(curr_err < best.err)
						std::cout << curr_err << " < " << best.err << std::endl;
					else
						std::cout << curr_err << " > " << best.err << std::endl;
				}
			}
		}
	}
	return best;
}

/*
 *	Function Name:	monte_carlo
 *	Description:	The function finds the best valid
 *					architecture, based on the input
 *					limitations, and returns it to the
 *					user.
 *					
 *	Parameters:
 *		uint32_t ni				Total number of inputs for the
 *								networks.
 *								
 *		uint32_t no				Total number of outputs for the
 *								networks.
 *								
 *		uint32_t ln				The highest number of possible
 *								nodes per hidden layer.
 *								
 *		uint32_t ll				The highest number of hidden la-
 *								yers allowed in a network.
 *								
 *		uint32_t num_bins		The number of bins to divide
 *								the training data into.
 *								
 *		double lr				The learning rate for the networks.
 *		
 *		double sf				The scaling factor for outputs in
 *								the networks.
 *								
 *		double dur 				The duration to train each network
 *								for.
 *								
 *		vector<double*> data	The data inputs and outputs for	|
 *								training.
 *								
 *	Returns:
 *		This function returns the network, its blueprint, its number
 *		of layers, and its error in a structure of type optarc_t.
 */
optarc_t monte_carlo(uint32_t ni, uint32_t no, uint32_t ln, uint32_t ll,
					 uint32_t num_bins, double lr, double sf, double dur,
					 std::vector<double*> data){
	// Initialize the random devices for random number generation.
	std::random_device rd_bin;
	std::random_device rd_ind;
	std::mt19937_64 gen_bin(rd_bin());
	std::mt19937_64 gen_ind(rd_ind());
	std::uniform_int_distribution<int> dist_bin(0, num_bins-1); // 0-based
	std::uniform_int_distribution<int> dist_ind(0, (data.size() - 1)); // 0-based
	// Random Number Generators Initialized!
	// Initialize the container for the bins.
	std::vector<std::vector<std::vector<double> > > bins;
	// bins
	/*
	 *	Outer vector holds the bins
	 *	Middle vector simulates the bins and holds vectors containing the data
	 *	Inner vector holds input and output data
	 */
	// Create the individual bins.
	for(unsigned int bin_counter = 0; bin_counter < num_bins; bin_counter++){
		std::vector<std::vector<double> > t_vect;
		bins.push_back(t_vect);
	}
	//unsigned int quantity_of_data = data.size(); // Get the quantity of information we are dividing.
	//unsigned int quant_per_bin = quantity_of_data / num_bins; // Find the amount of information each bin will store.
	// Split the data
	unsigned int bin_counter = 0; // Initialize a counter for bin.
	std::vector<unsigned int> visited; // visited list.
	// Loop until number of visited items is equivilent to the quantity of data we have.
	while(visited.size() < data.size()){
		unsigned int ind = (unsigned) dist_ind(gen_ind);
		// Iterate over the visited list
		bool v_flag = false;
		for(unsigned int visit_counter = 0; visit_counter < visited.size(); visit_counter++){
			if(visited[visit_counter] == ind){ // Is this index in the visited list
				v_flag = true; // Get new random index if visited before
				break;
			}
		}
		if(v_flag){
			continue;
		}
		visited.push_back(ind);
		// Create references for the inputs and the outputs.
		std::vector<double> data_ref;
		for(unsigned int data_counter = 0; data_counter < (ni + no); data_counter++){
			data_ref.push_back(data[ind][data_counter]);
		}
		// Push the vector holding the data values into the appropriate bin.
		bins[bin_counter].push_back(data_ref);
		// Increment bin_counter, if it is equal to num_bins, reinitialize it to 0.
		bin_counter++;
		bin_counter %= num_bins;
	}
	// Bins are now constructed with input data inside of them.
	// We should now create an object to store our best architecture thus far within.
	optarc_t best;
	// Time to create the initial blueprint. To make it simple,
	// it will just be the input and output layers.
	best.bp = std::vector<uint32_t>();
	best.nl = 0;
	best.err = 9999999999.99;
	// Initialize the network's structure.
	best.net_ref = 0;
	best.nn = 1;
	// We have an error now to compare other results with.
	// We need to construct the other blueprints now.
	std::vector<std::vector<unsigned int> > bps = construct_blueprints(ln, ll, ni, no);
	// Blueprints are now constructed.
	unsigned int num_threads = std::thread::hardware_concurrency(); // Find # of cores
	if(num_threads == 0) // Assume 1 core for systems w/out multiple cores
		num_threads = 1;
	uint32_t num_slices = bps.size() / num_threads;
	std::vector<shared_ptr<Network> > networks;
	for(uint32_t net_id; net_id < bps.size(); net_id++){
		uint32_t bp_size = bps[net_id].size();
		shared_ptr<Network> temp_ref(new Network(lr, bps[net_id], bp_size));
		networks.push_back(temp_ref);
	}
	#pragma omp parallel num_threads(num_threads) firstprivate(networks, dist_bin, gen_bin, bps, bins, dur, sf, lr, no, ni)
	{
		uint32_t my_rank = omp_get_thread_num();
		uint32_t my_si = my_rank * num_slices;
		uint32_t my_ei;
		if(my_rank == num_threads-1)
			my_ei = networks.size();
		else
			my_ei = my_si + num_slices;
		// Here we need to start the outer for loop, to iterate through every blueprint in bps.
		for(uint32_t i = my_si; i < my_ei; i++){
			double curr_err = 0.0;
			for(uint32_t bin_counter = 0; bin_counter < num_bins; bin_counter++){
				uint32_t rand_bin_num = dist_bin(gen_bin);
				std::vector<std::vector<double> > inputs;
				std::vector<std::vector<double> > outputs;
				for(uint32_t bc = 0 /*bc bincounter*/; bc < num_bins; bc++){
					if(bc == rand_bin_num) continue;
					for(uint32_t dc = 0; dc < bins[bc].size(); dc++){
						std::vector<double> vect_in(ni);
						std::vector<double> vect_out(no);
						for(uint32_t ic = 0; ic < ni; ic++){
							vect_in[ic] = bins[bc][dc][ic];
						}
						for(uint32_t oc = 0; oc < no; oc++){
							vect_out[oc] = bins[bc][dc][oc+ni];
						}
						inputs.push_back(vect_in);
						outputs.push_back(vect_out);
						//delete vect_in;
						//delete vect_out;
					}
				}
				networks[i]->train(inputs, outputs, sf, inputs.size(), dur);
				curr_err += test_network(networks[i], bins[rand_bin_num], ni, no, sf);
			}
			curr_err = (curr_err / num_bins) * 100;
			#pragma omp critical
			{
				uint32_t num_nodes = 0;
				for(uint32_t j = 0; j < bps[i].size(); j++){
					num_nodes += bps[i][j];
				}
				if((curr_err*num_nodes) < (best.err*best.nn)){
					std::cout << "Success\t\t";
					std::cout<< "[";
					for(uint32_t j = 0; j < bps[i].size();j++){
						std::cout << bps[i][j];
						if(j+1 != bps[i].size()) std::cout << ',';
					}
					if(bps[i].size() > 3)
						std::cout << "]\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					else
						std::cout << "]\t\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					if(curr_err < best.err)
						std::cout << curr_err << " < " << best.err << std::endl;
					else
						std::cout << curr_err << " > " << best.err << std::endl;
					//delete best.net_ref; // Free up pre-occupied memory.
					best.net_ref = networks[i]; // Store the reference to this network
					//delete best.bp; // Free up pre-occupied memory
					best.bp = bps[i];
					best.nl = bps[i].size();
					best.nn = num_nodes;
					best.err = curr_err; // Store this error.
				}
				else
				{
					std::cout << "Failure\t\t";
					std::cout<< "[";
					for(uint32_t j = 0; j < bps[i].size();j++){
						std::cout << bps[i][j];
						if(j+1 != bps[i].size()) std::cout << ',';
					}
					if(bps[i].size() > 3)
						std::cout << "]\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					else
						std::cout << "]\t\t" << (curr_err*num_nodes) << " < " << (best.err*best.nn) << "\t";
					if(curr_err < best.err)
						std::cout << curr_err << " < " << best.err << std::endl;
					else
						std::cout << curr_err << " > " << best.err << std::endl;
				}
			}
		}
	}
	return best;
}

double test_network(shared_ptr<Network> net,
					std::vector<std::vector<double> > data,
					uint32_t ni,
					uint32_t no,
					uint32_t sf){
	uint32_t num_data = data.size();
	std::vector<std::vector<double> > outputs_c; // outputs computed
	// Feed every input into the network.
	for(uint32_t i = 0; i < num_data; i++){
		std::vector<double> inputs; // Temp variable to hold inputs
		for(uint32_t j = 0; j < ni; j++){ // Store inputs into temp variable
			inputs.push_back(data[i][j]);
		}
		std::vector<double> outputs_ptr = net->feed(inputs); // Feed them
		outputs_c.push_back(outputs_ptr); // Push the output into the outputs vector.
	}
	std::vector<std::vector<double> > outputs_a; // outputs actual
	for(uint32_t i = 0; i < num_data; i++){
		std::vector<double> outputs;
		for(uint32_t j = 0; j < no; j++){
			outputs.push_back(data[i][j+ni]);
		}
		outputs_a.push_back(outputs);
	}
	// Time to compute error.
	double total_error = 0.0;
	for(uint32_t i = 0; i < num_data; i++){
		for(uint32_t j = 0; j < no; j++){
			double output_diff = outputs_a[i][j] - (outputs_c[i][j] * sf);
			double output_abs = abs(output_diff);
			total_error += (output_abs/outputs_a[i][j]);
		}
	}
	return total_error;
}
std::vector<std::vector<unsigned int> > construct_blueprints(uint32_t ln,
															  uint32_t ll,
															  uint32_t ni,
															  uint32_t no){
	std::vector<std::vector<unsigned int> > bps;
	for(uint32_t tl = 0; tl <= ll; tl++){
		std::vector<unsigned int> temp_var;
		// Push back the number of nodes a layer can have at most!
		for(uint32_t i = 0; i < tl; i++){
			temp_var.push_back(ln);
		}
		// total_bps_for_nl or total bluprints for number of layers
		uint32_t total_bps_for_nl = pow(ln, tl);
		// Create total_bps_for_nl blueprints
		for(uint32_t i = 0; i < total_bps_for_nl; i++){
			std::vector<unsigned int> bp(tl+2);
			// Set the first and last layers to be the number
			// of inputs, ni, and the number of outputs, no.
			bp[0] = ni;
			bp[tl+1] = no;
			// Set the rest of the contents in the bp to be the numbers
			for(uint32_t j = 1; j <= tl; j++){
				bp[j] = temp_var[j-1];
			}
			// If we are not on layer 0
			bool dec_flag = false;
			if(tl > 0){
				// Decrement a node in a layer
				uint32_t t_index = tl-1;
				temp_var[t_index]--;
				while(temp_var[t_index] == 0){
					if(t_index == 0 && temp_var[t_index] == 0){
						dec_flag = true;
						break;
					}
					temp_var[t_index] = ln;
					t_index--;
					temp_var[t_index]--;
				}
			}
			bps.push_back(bp);
			if(dec_flag) break;
		}
	}
	return bps;
}