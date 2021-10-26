#include <gubg/ann/Layer.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::Layer tests", "[ut][ann][Layer]")
{
	ann::Layer layer;

	layer.setup({.nr_inputs = 2, .nr_outputs = 3, .transfer = ann::Transfer::Tanh});


	ix::Range param_ixr;
	layer.setup_param_ixs(param_ixr);
	REQUIRE(param_ixr == ix::Range{0, 3*3});

	std::vector<float> params(param_ixr.size());
	param_ixr.each_with_index([](auto &v, auto ix){v = (ix+1)*0.1f;}, params);


	ix::Range input_ixr, output_ixr;
	layer.setup_io_ixs(input_ixr, output_ixr);
	REQUIRE(input_ixr == ix::Range{0, 2});
	REQUIRE(output_ixr == ix::Range{2, 3});

	std::vector<float> activations(output_ixr.end());
	activations[0] = 1.0f;
	activations[1] = 2.0f;

	std::vector<float> sufficients(output_ixr.end());

	layer.forward(params, activations, sufficients);
	REQUIRE(activations[output_ixr[0]] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
	REQUIRE(activations[output_ixr[1]] == Approx(ann::transfer::Tanh::output(0.4 + 1*0.5 + 2*0.6)));
	REQUIRE(activations[output_ixr[2]] == Approx(ann::transfer::Tanh::output(0.7 + 1*0.8 + 2*0.9)));
}	
