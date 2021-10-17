#include <gubg/ann/MLP.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::MLP tests", "[ut][ann][MLP]")
{
	ann::MLP mlp;

	mlp.setup({.nr_inputs = 2});
	mlp.add_layer(3, ann::Transfer::Tanh);
	mlp.add_layer(1, ann::Transfer::Linear);

	std::size_t nr_inputs = 0u;
	std::size_t nr_outputs = 0u;
	mlp.consume_io(nr_inputs, nr_outputs);
	REQUIRE(nr_inputs == 2);
	REQUIRE(nr_outputs == 4);

	std::size_t nr_params = 0u;
	mlp.consume_params(nr_params);
	REQUIRE(nr_params == 3*3+4);

	std::vector<float> inputs(nr_inputs);
	inputs[0] = 1.0f;
	inputs[1] = 2.0f;

	std::vector<float> params(nr_params);
	for (auto ix = 0u; ix < nr_params; ++ix)
		params[ix] = (ix+1)*0.1f;

	std::vector<float> outputs(nr_outputs);

	mlp.forward(inputs, params, outputs);
	REQUIRE(outputs[0] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
	REQUIRE(outputs[1] == Approx(ann::transfer::Tanh::output(0.4 + 1*0.5 + 2*0.6)));
	REQUIRE(outputs[2] == Approx(ann::transfer::Tanh::output(0.7 + 1*0.8 + 2*0.9)));
	REQUIRE(outputs[3] == Approx(params[9] + outputs[0]*params[10]+outputs[1]*params[11]+outputs[2]*params[12]));
}	
