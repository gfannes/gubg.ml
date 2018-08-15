#include "catch.hpp"
#include "gubg/neural/Trainer.hpp"
#include "gubg/hr.hpp"
#include "gubg/naft/Document.hpp"
#include <cmath>
#include <iostream>
#include <random>
using namespace gubg;

namespace  { 
    using Float = float;
    using Simulator = neural::Simulator<Float>;
    using Trainer = neural::Trainer<Float>;
    using Vector = std::vector<Float>;
} 

TEST_CASE("neural::Trainer tests", "[ut][neural][Trainer]")
{
    //Create the trainer, ready to train a network with 1 input and 1 output
    Trainer trainer(1,1);

    //Add the training data: sine data
    for (Float x = -2.0; x <= 2.0; x += 0.01)
    {
        REQUIRE(!trainer.add(Vector{x,x}, Vector()));
        REQUIRE(trainer.add(Vector{x}, Vector{std::sin(x)}));
    }
    REQUIRE(trainer.data_size() == 401);

    //Create the neural::Simulator
    Simulator simulator;
    const auto input = simulator.add_external(1);
    const auto bias = simulator.add_external(1);
    Vector layer_inputs = {Float(input), bias};
    auto add_layer = [&](int nr){
        Vector layer_outputs;
        for (int i = 0; i < nr; ++i)
        {
            size_t output;
            simulator.add_neuron(neural::Transfer::Tanh, layer_inputs, output);
            layer_outputs.push_back(output);
        }
        layer_inputs.swap(layer_outputs);
    };
#if 1
    //2 hidden layers with 10 tanh neurouns
    //1 linear output neuron
    for (int i = 0; i < 2; ++i)
        add_layer(10);
    size_t output;
    simulator.add_neuron(neural::Transfer::Linear, layer_inputs, output);
#else
    //1 tanh neuron
    size_t output;
    simulator.add_neuron(neural::Transfer::Tanh, layer_inputs, output);
#endif

    //Inject this neural::Simulator into the trainer
    REQUIRE(!trainer.set(nullptr, input, output));
    REQUIRE(trainer.set(&simulator, input, output));
    trainer.add_fixed_input(bias, 1.0);

    //Generate randomized weights
    Vector weights(simulator.nr_weights());
    std::mt19937 eng;
    std::normal_distribution<Float> normal{0.0, 0.1};
    std::generate(RANGE(weights), [&](){return normal(eng);});

    naft::Document doc(std::cout);

    const unsigned int nr_steps = 200;
    const Float output_stddev = 0.1;
    const Float weights_stddev = 10.0;

    SECTION("steepest descent")
    {
        auto root = doc.node("sd");
        double lp = 0.0;
        Float step = 0.0015;
        for (int i = 0; i < nr_steps; ++i)
        {
            double newlp;
            REQUIRE(trainer.train_sd(newlp, weights.data(), output_stddev, weights_stddev, step));
            step *= (newlp < lp ? 0.5 : 1.01);
            lp = newlp;
            root.node("train").attr("lp",lp).attr("step",step);
        }

        if (false)
        {
            Vector states(simulator.nr_states());
            states[bias] = 1.0;
            for (Float x = -2.0; x <= 2.0; x += 0.01)
            {
                states[input] = x;
                simulator.forward(states.data(), weights.data());
                const auto o = states[output];
                root.node("point").attr("x",x).attr("t",std::sin(x)).attr("o",o);
            }
        }
    }
    SECTION("adam")
    {
        auto root = doc.node("adam");
        double lp = 0.0;
        Trainer::AdamParams params;
        params.alpha = 0.01;
        params.beta1 = 0.9;
        REQUIRE(trainer.init_adam(params));
        for (int i = 0; i < nr_steps; ++i)
        {
            REQUIRE(trainer.train_adam(lp, weights.data(), output_stddev, weights_stddev));
            root.node("train").attr("lp",lp);
        }

        if (false)
        {
            Vector states(simulator.nr_states());
            states[bias] = 1.0;
            for (Float x = -2.0; x <= 2.0; x += 0.01)
            {
                states[input] = x;
                simulator.forward(states.data(), weights.data());
                const auto o = states[output];
                root.node("point").attr("x",x).attr("t",std::sin(x)).attr("o",o);
            }
        }
    }
    SECTION("scg")
    {
        auto root = doc.node("scg");
        double lp = 0.0;
        for (int i = 0; i < nr_steps; ++i)
        {
            REQUIRE(trainer.train_scg(lp, weights.data(), output_stddev, weights_stddev, 1));
            root.node("train").attr("lp",lp);
        }

        if (false)
        {
            Vector states(simulator.nr_states());
            states[bias] = 1.0;
            for (Float x = -2.0; x <= 2.0; x += 0.01)
            {
                states[input] = x;
                simulator.forward(states.data(), weights.data());
                const auto o = states[output];
                root.node("point").attr("x",x).attr("t",std::sin(x)).attr("o",o);
            }
        }
    }
}
