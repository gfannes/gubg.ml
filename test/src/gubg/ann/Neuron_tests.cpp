#include <gubg/ann/Neuron.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg;

TEST_CASE("ANN Neuron tests", "[ut][ann][Neuron]")
{
	ann::Neuron neuron;

	neuron.setup({.nr_inputs = 2, .transfer = ann::Transfer::Tanh});

	std::size_t nr_inputs = 0u;
	std::size_t nr_outputs = 0u;
	neuron.consume_io(nr_inputs, nr_outputs);
	REQUIRE(nr_inputs == 2);
	REQUIRE(nr_outputs == 1);

	std::size_t nr_params = 0u;
	neuron.consume_params(nr_params);
	REQUIRE(nr_params == 3);

	std::vector<float> inputs(nr_inputs);
	inputs[0] = 1.0f;
	inputs[1] = 2.0f;

	std::vector<float> params(nr_params);
	params[0] = 0.1f;//bias
	params[1] = 0.2f;
	params[2] = 0.3f;

	std::vector<float> outputs(nr_outputs);

	neuron.forward(inputs, params, outputs);
	REQUIRE(outputs[0] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
}	
