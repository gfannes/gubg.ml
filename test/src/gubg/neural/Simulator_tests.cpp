#include "catch.hpp"
#include "gubg/neural/Simulator.hpp"
#include "gubg/naft/Document.hpp"
#include "gubg/hr.hpp"
#include <vector>
#include <iostream>
#include <numeric>
using namespace gubg;

namespace  { 
    using Float = double;
    using Vector = std::vector<Float>;
    using Inputs = std::vector<size_t>;
} 

TEST_CASE("neural::Simulator tests", "[ut][neural][Simulator]")
{
    neural::Simulator<Float> simulator;

    REQUIRE(simulator.nr_states() == 0);
    REQUIRE(simulator.nr_weights() == 0);

    const auto input = simulator.add_external(1);
    const auto bias = simulator.add_external(1);

    const auto inputs = Inputs{input, bias};

    SECTION("single neuron of each type")
    {
        struct Info
        {
            size_t output_ix, weight_ix;
            Float output;
        };
        Info linear, tanh, sigmoid, leakyrelu, quadratic;
        REQUIRE(simulator.add_neuron(neural::Transfer::Linear,    inputs, linear.output_ix,    linear.weight_ix));
        REQUIRE(simulator.add_neuron(neural::Transfer::Tanh,      inputs, tanh.output_ix,      tanh.weight_ix));
        REQUIRE(simulator.add_neuron(neural::Transfer::Sigmoid,   inputs, sigmoid.output_ix,   sigmoid.weight_ix));
        REQUIRE(simulator.add_neuron(neural::Transfer::LeakyReLU, inputs, leakyrelu.output_ix, leakyrelu.weight_ix));
        REQUIRE(simulator.add_neuron(neural::Transfer::Quadratic, inputs, quadratic.output_ix, quadratic.weight_ix));

        REQUIRE(simulator.nr_weights() == 2+2+2+2+2);

        Vector weights(simulator.nr_weights());
        weights[linear.weight_ix] = 1.0;
        weights[tanh.weight_ix] = 1.0;
        weights[sigmoid.weight_ix] = 10.0;
        weights[sigmoid.weight_ix+1] = -10.0;
        weights[leakyrelu.weight_ix] = 1.0;
        weights[quadratic.weight_ix] = 1.0;

        Vector states(simulator.nr_states());
        states[bias] = 1.0;
        Vector states2(simulator.nr_states());
        states2[bias] = 1.0;
        Vector preacts(simulator.nr_states());

        naft::Document naft(std::cout);
        auto root = naft.node("simulator");
        for (Float x = -2.0; x <= 2.0; x += 0.01)
        {
            states[input] = x;
            simulator.forward(states.data(), preacts.data(), weights.data());
            states2[input] = x;
            simulator.forward(states2.data(), weights.data());

            REQUIRE(states == states2);

            linear.output = states[linear.output_ix];
            REQUIRE(x == preacts[linear.output_ix]);
            REQUIRE(x == linear.output);

            tanh.output = states[tanh.output_ix];
            REQUIRE((x < 0.0)  == (tanh.output < 0.0));
            REQUIRE((x == 0.0) == (tanh.output == 0.0));
            REQUIRE((x > 0.0)  == (tanh.output > 0.0));
            REQUIRE(x == preacts[tanh.output_ix]);

            sigmoid.output = states[sigmoid.output_ix];
            REQUIRE((x < 1.0)  == (sigmoid.output < 0.5));
            REQUIRE((x == 1.0) == (sigmoid.output == 0.5));
            REQUIRE((x > 1.0)  == (sigmoid.output > 0.5));
            REQUIRE(preacts[sigmoid.output_ix] == Approx(x*10.0-10.0));

            leakyrelu.output = states[leakyrelu.output_ix];
            REQUIRE(((x < 0.0 && x < leakyrelu.output) || (x >= 0.0 && x == leakyrelu.output)));
            REQUIRE(x == preacts[leakyrelu.output_ix]);

            quadratic.output = states[quadratic.output_ix];
            REQUIRE((x == 0.0) == (quadratic.output == 0.0));
            REQUIRE(quadratic.output >= 0.0);

            root.node("point").attr("x", x)
            .attr("linear", linear.output)
            .attr("tanh", tanh.output)
            .attr("sigmoid", sigmoid.output)
            .attr("leakyrelu", leakyrelu.output)
            .attr("quadratic", quadratic.output);
        }
    }

    SECTION("many neurons in many layers")
    {
        size_t layer_input = input;
        auto add_layer = [&](int nr_input, int nr_output)
        {
            bool ok = true;
            Inputs inputs(nr_input+1);
            std::iota(RANGE(inputs), layer_input);
            for (unsigned int i = 0; i < nr_output; ++i)
            {
                size_t output;
                ok && (ok = simulator.add_neuron(neural::Transfer::Tanh, inputs, output));
                if (i == 0)
                    layer_input = output;
            }
            return ok;
        };

        const int size = 500;
        const int nr_layers = 50;
        REQUIRE(add_layer(1, size));
        for (auto i = 0; i < nr_layers; ++i)
            REQUIRE(add_layer(size, size));

        std::cout << C(simulator.nr_weights())C(simulator.nr_states()) << std::endl;
        Vector weights(simulator.nr_weights());
        std::fill(RANGE(weights), 0.001);
        Vector states(simulator.nr_states());
        const Float nan = 999.9;
        std::fill(RANGE(states), nan);
        states[bias] = 1.0;

        states[input] = 1.0;
        for (int i = 0; i < 10; ++i)
            simulator.forward(states.data(), weights.data());
        REQUIRE(states[layer_input] != nan);
        //The weights are chosen to not saturate with current layer size, nor cause 0-output
        REQUIRE(0.0 < states[layer_input]);
        REQUIRE(states[layer_input] < 1.0);
    }

    SECTION("loglikelihood")
    {
        const auto output = simulator.add_external(1);

        size_t ll = 999;
        std::array<size_t, 1> inputs = {input}, outputs = {output};
        REQUIRE(simulator.add_loglikelihood(inputs, outputs, ll, 0.1));
        REQUIRE(ll != 999);
        REQUIRE(simulator.nr_weights() == 0);

        Vector states(simulator.nr_states());
        states[input] = 1.0;
        states[output] = 3.0;

        simulator.forward(states.data(), nullptr);
        REQUIRE(states[ll] == 200);

        std::cout << C(states[ll]) << std::endl;
    }

    SECTION("backward")
    {
        size_t output, weight;
        simulator.add_neuron(neural::Transfer::Quadratic, inputs, output, weight);

        Vector states(simulator.nr_states());
        states[input] = 2.0;
        states[bias] = 1.0;

        Vector weights(simulator.nr_weights());
        weights[weight] = 0.1;
        weights[weight+1] = 0.3;

        Vector preacts(simulator.nr_states());
        simulator.forward(states.data(), preacts.data(), weights.data());

        Vector derivative(simulator.nr_states());
        derivative[output] = 1.0;
        Vector gradient(simulator.nr_weights());
        simulator.backward(derivative.data(), gradient.data(), states.data(), preacts.data(), weights.data());

        std::cout << C(gubg::hr(derivative)) << std::endl;
        std::cout << C(gubg::hr(gradient)) << std::endl;
        REQUIRE((derivative == Vector{Float(0.05), Float(0.15), Float(0.5)}));
        REQUIRE((gradient == Vector{Float(1.0), 0.5}));
    }
}
