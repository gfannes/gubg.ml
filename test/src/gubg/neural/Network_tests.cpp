#include "catch.hpp"
#include "gubg/neural/Network.hpp"
#include "gubg/naft/Document.hpp"
#include <vector>
#include <iostream>
using namespace gubg;

namespace  { 
    using Float = float;
    using Vector = std::vector<Float>;
    using Inputs = std::vector<size_t>;
} 

TEST_CASE("neural::Network tests", "[ut][neural]")
{
    neural::Network<Float> nn(1);

    REQUIRE(nn.nr_weights() == 0);
    REQUIRE(nn.nr_postacts() == 1+1);

    SECTION("forward without any neuron")
    {
        Vector weights(nn.nr_weights());
        Vector postacts(nn.nr_postacts());
        postacts[nn.input(0)] = 0.0;
        nn.forward(postacts.data(), weights.data());
    }

    SECTION("single neuron of each type")
    {
        const auto inputs = Inputs{nn.input(0), nn.bias()};
        struct Info
        {
            size_t output_ix, weight_ix;
            Float output;
        };
        Info linear, tanh, sigmoid, leakyrelu;
        REQUIRE(nn.add_neuron(neural::Transfer::Linear,    inputs, &linear.output_ix, &linear.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::Tanh,      inputs, &tanh.output_ix, &tanh.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::Sigmoid,   inputs, &sigmoid.output_ix, &sigmoid.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::LeakyReLU, inputs, &leakyrelu.output_ix, &leakyrelu.weight_ix));

        REQUIRE(nn.nr_weights() == 2+2+2+2);

        Vector weights(nn.nr_weights());
        weights[linear.weight_ix] = 1.0;
        weights[tanh.weight_ix] = 1.0;
        weights[sigmoid.weight_ix] = 10.0;
        weights[sigmoid.weight_ix+1] = -10.0;
        weights[leakyrelu.weight_ix] = 1.0;

        Vector postacts(nn.nr_postacts());

        naft::Document naft(std::cout);
        auto root = naft.node("tanh");
        for (Float x = -2.0; x <= 2.0; x += 0.01)
        {
            postacts[nn.input(0)] = x;
            nn.forward(postacts.data(), weights.data());

            linear.output = postacts[linear.output_ix];
            REQUIRE(x == linear.output);

            tanh.output = postacts[tanh.output_ix];
            REQUIRE((x < 0.0)  == (tanh.output < 0.0));
            REQUIRE((x == 0.0) == (tanh.output == 0.0));
            REQUIRE((x > 0.0)  == (tanh.output > 0.0));

            sigmoid.output = postacts[sigmoid.output_ix];
            REQUIRE((x < 1.0)  == (sigmoid.output < 0.5));
            REQUIRE((x == 1.0) == (sigmoid.output == 0.5));
            REQUIRE((x > 1.0)  == (sigmoid.output > 0.5));

            leakyrelu.output = postacts[leakyrelu.output_ix];
            REQUIRE(((x < 0.0 && x < leakyrelu.output) || (x >= 0.0 && x == leakyrelu.output)));

            root.node("point").attr("x", x).attr("linear", linear.output).attr("tanh", tanh.output).attr("sigmoid", sigmoid.output).attr("leakyrelu", leakyrelu.output);
        }
    }
}
