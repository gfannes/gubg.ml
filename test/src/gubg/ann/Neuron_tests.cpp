#include <gubg/ann/Neuron.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg;

TEST_CASE("ANN Neuron tests", "[ut][ann][Neuron]")
{
	ann::Neuron neuron;

	neuron.setup({.nr_inputs = 2, .transfer = ann::Transfer::Tanh});


	ix::Range param_ixr;
	neuron.setup_param_ixs(param_ixr);
	REQUIRE(param_ixr == ix::Range{0, 3});

	std::vector<float> params(param_ixr.end());
	params[0] = 0.1f;//bias
	params[1] = 0.2f;
	params[2] = 0.3f;

	ix::Range input_ixr, output_ixr;
	neuron.setup_io_ixs(input_ixr, output_ixr);
	REQUIRE(input_ixr == ix::Range{0, 2});
	REQUIRE(output_ixr == ix::Range{2, 1});

	std::vector<float> activations(output_ixr.end());
	activations[0] = 1.0f;
	activations[1] = 2.0f;


	std::vector<float> sufficients(output_ixr.end());

	neuron.forward(params, activations, sufficients);
	REQUIRE(activations[output_ixr[0]] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
}	
