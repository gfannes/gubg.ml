#include <gubg/ann/Model.hpp>
#include <catch.hpp>
using namespace gubg;

TEST_CASE("ann::Model tests", "[ut][ann][Model]")
{
	ann::Model model;

	model.prediction_cost.quadratic(0.1);

	double cost;
	REQUIRE(!model.avg_prediction_cost(cost, [](){}));
}	