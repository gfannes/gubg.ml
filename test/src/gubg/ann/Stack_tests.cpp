#include <gubg/ann/Stack.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::Stack tests", "[ut][ann][Stack]")
{
	ann::Stack stack;


	stack.setup({.nr_inputs = 2});
	stack.add_layer(3, ann::Transfer::Tanh);
	stack.add_layer(1, ann::Transfer::Linear);


	ix::Range input_ixr;
	ix::Range_opt output_ixr_opt;
	ix::Range param_ixr;
	REQUIRE(stack.setup_ixrs(input_ixr, output_ixr_opt, param_ixr));

	REQUIRE(input_ixr == ix::Range{0,2});
	REQUIRE(!!output_ixr_opt);
	const auto &output_ixr = *output_ixr_opt;
	REQUIRE(output_ixr == ix::Range{5,1});
	REQUIRE(param_ixr == ix::Range{0, 3*3+4});

	ix::Range hidden_ixr{input_ixr.end(), 3};

	std::vector<float> activations(output_ixr.end());
	input_ixr.each_with_index([](auto &v, auto ix){v = ix+1.0f;}, activations);

	std::vector<float> params(param_ixr.size());
	param_ixr.each_with_index([](auto &v, auto ix){v = (ix+1)*0.1f;}, params);

	std::vector<float> sufficients(output_ixr.end());

	stack.forward(params, activations, sufficients);
	REQUIRE(activations[hidden_ixr[0]] == Approx(ann::transfer::Tanh::output(0.1 + 1*0.2 + 2*0.3)));
	REQUIRE(activations[hidden_ixr[1]] == Approx(ann::transfer::Tanh::output(0.4 + 1*0.5 + 2*0.6)));
	REQUIRE(activations[hidden_ixr[2]] == Approx(ann::transfer::Tanh::output(0.7 + 1*0.8 + 2*0.9)));
	REQUIRE(activations[output_ixr[0]] == Approx(params[9] + activations[hidden_ixr[0]]*params[10]+activations[hidden_ixr[1]]*params[11]+activations[hidden_ixr[2]]*params[12]));
}	
