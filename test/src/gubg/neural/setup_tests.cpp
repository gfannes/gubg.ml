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

    SECTION("setup structure from mlp::Structure")
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

        SECTION("setup parameters from mlp::Parameters")
        {
            mlp::Parameters p;
            p.setup_from(s);

            Vector weights(simulator.nr_weights(), 999.9);

            SECTION("correct amount of weights")
            {
                REQUIRE(neural::setup(weights, p));
                for (auto v: weights)
                {
                    REQUIRE(v == 0.0);
                }
            }
            SECTION("not enough weights")
            {
                weights.pop_back();
                REQUIRE(!neural::setup(weights, p));
            }
            SECTION("too much weights")
            {
                weights.push_back(1.0);
                REQUIRE(!neural::setup(weights, p));
            }
        }
    }
}
