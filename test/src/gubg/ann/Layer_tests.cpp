#include <gubg/ann/Layer.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::Layer tests", "[ut][ann][Layer]")
{
	ann::Layer layer;

	layer.setup({.nr_inputs = 2, .nr_outputs = 3, .transfer = ann::Transfer::Tanh});

	std::size_t nr_inputs = 0u;
	std::size_t nr_outputs = 0u;
	layer.consume_io(nr_inputs, nr_outputs);
	REQUIRE(nr_inputs == 2);
	REQUIRE(nr_outputs == 3);

	std::size_t nr_params = 0u;
	layer.consume_params(nr_params);
	REQUIRE(nr_params == 3*3);

	std::vector<float> inputs(nr_inputs);
	inputs[0] = 1.0f;
	inputs[1] = 2.0f;

	std::vector<float> params(nr_params);
	for (auto ix = 0u; ix < nr_params; ++ix)
		params[ix] = (ix+1)*0.1f;

	std::vector<float> outputs(nr_outputs);

	layer.forward(inputs, params, outputs);
	REQUIRE(outputs[0] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
	REQUIRE(outputs[1] == Approx(ann::transfer::Tanh::output(0.4 + 1*0.5 + 2*0.6)));
	REQUIRE(outputs[2] == Approx(ann::transfer::Tanh::output(0.7 + 1*0.8 + 2*0.9)));
}	
