#include <gubg/ann/Model.hpp>
#include <gubg/hr.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg;

TEST_CASE("ann::Model tests", "[ut][ann][Model]")
{
	S("");

	const auto cost_sigma = 0.1;
	const auto prediction = 0.5;

	const float input = 2.0f;
	float target{};
	float exp_cost{};
	std::vector<float> exp_gradient;
	SECTION("target == prediction")
	{
		target = prediction;
		exp_cost = 0.0f;
		exp_gradient = {0.0f, 0.0f};
	}
	SECTION("target == 1")
	{
		target = 1.0f;
		exp_cost = (prediction-target)*(prediction-target)/2/cost_sigma/cost_sigma;
		exp_gradient = {-12.5f, -25.0f};
	}

	ann::Model model;

	REQUIRE(model.update_stack([&](auto &stack){
		stack.setup(ann::Stack::Shape{.nr_inputs = 1, .nr_outputs = 1});
		stack.add_layer(1, ann::Transfer::Sigmoid);
	}));

	model.prediction_cost.quadratic(cost_sigma);

	auto enter_no_data = [](auto &&entry){};
	auto enter_1_data = [&](auto &&entry){
		L(C(input)C(target));
		entry(&input, &target);
	};

	double act_cost = 0.0;
	REQUIRE(!model.avg_cost(act_cost, enter_no_data));
	REQUIRE(model.avg_cost(act_cost, enter_1_data));
	L(C(act_cost));
	REQUIRE(act_cost == Approx(exp_cost));

	REQUIRE(!model.avg_gradient(enter_no_data));
	REQUIRE(model.avg_gradient(enter_1_data));
	REQUIRE(model.gradient.size() == exp_gradient.size());
	for (auto ix = 0u; ix < model.gradient.size(); ++ix)
		REQUIRE(model.gradient[ix] == Approx(exp_gradient[ix]));
}	