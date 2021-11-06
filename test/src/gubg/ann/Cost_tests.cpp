#include <gubg/ann/Cost.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg;

TEST_CASE("ann::Cost tests", "[ut][ann][Cost]")
{
	ann::Cost cost;
	cost.quadratic(0.1);

	std::vector<float> prediction = {0.9f, 2.2f};
	std::vector<float> target = {1.0f, 2.0f};

	ix::Range prediction_ixr{0u, prediction.size()};
	ix::Range target_ixr{0u, target.size()};

	REQUIRE(cost.setup_ixrs(prediction_ixr, target_ixr));

	double total_cost = 0;
	REQUIRE(cost.add_cost(total_cost, prediction, target));

	auto diff = 0.0;
	diff += std::pow(prediction[0]-target[0], 2);
	diff += std::pow(prediction[1]-target[1], 2);
	REQUIRE(total_cost == Approx(diff/2.0/0.1/0.1));
}	