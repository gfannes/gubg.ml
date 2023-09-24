#include <gubg/ann/Layer.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::Layer tests", "[ut][ann][Layer]")
{
	ann::Layer layer;

	layer.setup({.nr_inputs = 2, .nr_outputs = 3, .transfer = ann::Transfer::Tanh});


	ix::Range input_ixr;
	ix::Range_opt output_ixr_opt;
	ix::Range param_ixr;
	REQUIRE(layer.setup_ixrs(input_ixr, output_ixr_opt, param_ixr));

	REQUIRE(input_ixr == ix::Range{0, 2});
	REQUIRE(!!output_ixr_opt);
	const auto &output_ixr = *output_ixr_opt;
	REQUIRE(output_ixr == ix::Range{2, 3});
	REQUIRE(param_ixr == ix::Range{0, 3*3});

	std::vector<float> activations(output_ixr.stop());
	input_ixr.each_with_index([](auto &v, auto ix){v = ix+1.0f;}, activations);

	std::vector<float> params(param_ixr.size());
	param_ixr.each_with_index([](auto &v, auto ix){v = (ix+1)*0.1f;}, params);

	std::vector<float> sufficients(output_ixr.stop());

	layer.forward(params, activations, sufficients);
	REQUIRE(activations[output_ixr[0]] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
	REQUIRE(activations[output_ixr[1]] == Approx(ann::transfer::Tanh::output(0.4 + 1*0.5 + 2*0.6)));
	REQUIRE(activations[output_ixr[2]] == Approx(ann::transfer::Tanh::output(0.7 + 1*0.8 + 2*0.9)));
}	
