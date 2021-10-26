#include <gubg/ann/Neuron.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg;

TEST_CASE("ANN Neuron tests", "[ut][ann][Neuron]")
{
	ann::Neuron neuron;

	neuron.setup({.nr_inputs = 2, .transfer = ann::Transfer::Tanh});


	ix::Range input_ixr;
	ix::Range_opt output_ixr_opt;
	ix::Range param_ixr;
	REQUIRE(neuron.setup_ixrs(input_ixr, output_ixr_opt, param_ixr));

	REQUIRE(input_ixr == ix::Range{0, 2});
	REQUIRE(!!output_ixr_opt);
	const auto &output_ixr = *output_ixr_opt;
	REQUIRE(output_ixr == ix::Range{2, 1});
	REQUIRE(param_ixr == ix::Range{0, 3});

	std::vector<float> activations(output_ixr.end());
	input_ixr.each_with_index([](auto &v, auto ix){v = ix+1.0f;}, activations);

	std::vector<float> params(param_ixr.end());
	param_ixr.each_with_index([](auto &v, auto ix){v = (ix+1)*0.1f;}, params);

	std::vector<float> sufficients(output_ixr.end());

	neuron.forward(params, activations, sufficients);
	REQUIRE(activations[output_ixr[0]] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
}	
