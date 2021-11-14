#include <gubg/ann/optimization.hpp>
#include <catch.hpp>
#include <vector>

#include <gubg/debug.hpp>
#include <gubg/hr.hpp>

using namespace gubg;
using namespace gubg::ann;

TEST_CASE("ann::optimization::SteepestDescent tests", "[ut][ann][optimization][SteepestDescent]")
{
	S("");

	optimization::SteepestDescent sd;
	REQUIRE(sd.params.learning_rate == 0.01f);

	std::vector<Float> location = {0.0f, 0.0f};
	std::vector<Float> gradient = {0.1f, 0.2f};
	REQUIRE(!sd.update(location, gradient));

	const auto ixr = ix::Range{0, location.size()};
	sd.setup_ixr(ixr);

	for (auto i = 1; i < 10; ++i)
	{
		REQUIRE(sd.update(location, gradient));
		ixr.each(
			[&](auto loc, auto grad){REQUIRE(loc == Approx(i*sd.params.learning_rate*grad));},
			location, gradient
			);
	}
}	