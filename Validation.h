#ifndef VALIDATION_H_
#define VALIDATION_H_

#include <stdint.h>
#include "Network.h"
#include <vector>
#include <memory>

using namespace std;
using namespace AeroSW;

/*
 *	Structure:	Optimal_Architecture
 *	Purpose:	Store basic architecture of a
 *				NEURAL NETWORK and information
 *				about the network.
 *	Items: net_ref -->	Pointer Reference to
 *							a Neural Network.
 *				bp -->	Array of unsigned integers
 *						used to store the layout
 *						of the network.
 *				nl -->	Unsigned integer used to
 *						specify how many items are
 *						in the bp.
 *			   err -->	Total error this architecture
 *						contained after training.
 */
struct optarc_t{
	shared_ptr<Network> net_ref; // network reference
	std::vector<uint32_t> bp; // blueprint
	uint32_t nl; // num layers
	uint32_t nn; // num nodes
	double err; // error
};

optarc_t k_fold_cross(uint32_t ni, uint32_t no, uint32_t ln, uint32_t ll, uint32_t num_bins, double lr, double sf, double dur, std::vector<double*> data);
optarc_t monte_carlo(uint32_t ni, uint32_t no, uint32_t ln, uint32_t ll, uint32_t num_bins, double lr, double sf, double dur, std::vector<double*> data);
double test_network(std::shared_ptr<Network> net, std::vector<std::vector<double> > data, uint32_t ni, uint32_t no, uint32_t sf);
std::vector<std::vector<unsigned int> > construct_blueprints(uint32_t ln, uint32_t ll, uint32_t ni, uint32_t no);

#endif
