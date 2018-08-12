#include "catch.hpp"
#include "gubg/neural/setup.hpp"
using namespace gubg;

namespace  { 
    using Float = double;
    using Vector = std::vector<Float>;
    using Inputs = std::vector<size_t>;
} 

TEST_CASE("neural::setup tests", "[ut][neural][setup]")
{
    neural::Simulator<Float> simulator;

    SECTION("setup from mlp::Structure")
    {
        mlp::Structure s(2);
        s.add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
        s.add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
        size_t first_input, bias, first_output;
        REQUIRE(neural::setup(simulator, s, first_input, bias, first_output));
        REQUIRE(simulator.nr_states() == (2+1+5+1));
        REQUIRE(simulator.nr_weights() == (5*3+1*6));
        REQUIRE(first_input == 0);
        REQUIRE(bias == 2);
        REQUIRE(first_output == 8);
    }
}
