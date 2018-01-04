#include "catch.hpp"
#include "gubg/neural/Network.hpp"
#include "gubg/naft/Document.hpp"
#include <vector>
#include <iostream>
#include <numeric>
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

    const auto inputs = Inputs{nn.input(0), nn.bias()};

    SECTION("single neuron of each type")
    {
        struct Info
        {
            size_t output_ix, weight_ix;
            Float output;
        };
        Info linear, tanh, sigmoid, leakyrelu;
        REQUIRE(nn.add_neuron(neural::Transfer::Linear,    inputs, linear.output_ix, linear.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::Tanh,      inputs, tanh.output_ix, tanh.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::Sigmoid,   inputs, sigmoid.output_ix, sigmoid.weight_ix));
        REQUIRE(nn.add_neuron(neural::Transfer::LeakyReLU, inputs, leakyrelu.output_ix, leakyrelu.weight_ix));

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

    SECTION("many neurons in many layers")
    {
        size_t input = 0;
        auto add_layer = [&](int nr_input, int nr_output)
        {
            bool ok = true;
            Inputs inputs(nr_input+1);
            std::iota(RANGE(inputs), input);
            for (unsigned int i = 0; i < nr_output; ++i)
            {
                size_t output;
                ok && (ok = nn.add_neuron(neural::Transfer::Tanh, inputs, output));
                if (i == 0)
                    input = output;
            }
            return ok;
        };

        const int size = 500;
        const int nr_layers = 50;
        REQUIRE(add_layer(1, size));
        for (auto i = 0; i < nr_layers; ++i)
            REQUIRE(add_layer(size, size));

        std::cout << C(nn.nr_weights())C(nn.nr_postacts()) << std::endl;
        Vector weights(nn.nr_weights());
        std::fill(RANGE(weights), 0.001);
        Vector postacts(nn.nr_postacts());
        const Float nan = 999.9;
        std::fill(RANGE(postacts), nan);

        postacts[nn.input(0)] = 1.0;
        for (int i = 0; i < 10; ++i)
            nn.forward(postacts.data(), weights.data());
        REQUIRE(postacts[input] != nan);
        //The weights are chosen to not saturate with current layer size, nor cause 0-output
        REQUIRE(0.0 < postacts[input]);
        REQUIRE(postacts[input] < 1.0);
    }
}
