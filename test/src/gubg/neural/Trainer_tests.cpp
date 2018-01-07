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
    using Network = neural::Network<Float>;
    using Trainer = neural::Trainer<Float>;
    using Vector = std::vector<Float>;
} 

TEST_CASE("neural::Trainer tests", "[ut][neural][Trainer]")
{
    Trainer trainer(1,1);

    for (Float x = -2.0; x <= 2.0; x += 0.01)
    {
        REQUIRE(!trainer.add(Vector{x,x}, Vector()));
        REQUIRE(trainer.add(Vector{x}, Vector{std::sin(x)}));
    }
    REQUIRE(trainer.data_size() == 401);

    Network nn;
    const auto input = nn.add_external(1);
    const auto bias = nn.add_external(1);
    Vector layer_outputs;
    for (int i = 0; i < 100; ++i)
    {
        size_t output;
        nn.add_neuron(neural::Transfer::Tanh, Vector{input, bias}, output);
        layer_outputs.push_back(output);
    }
    size_t output;
    nn.add_neuron(neural::Transfer::Linear, layer_outputs, output);

    REQUIRE(!trainer.set(nullptr, input, output));
    REQUIRE(trainer.set(&nn, input, output));
    trainer.add_fixed_input(bias, 1.0);

    Vector weights(nn.nr_weights());
    std::mt19937 eng;
    std::normal_distribution<Float> normal;
    std::generate(RANGE(weights), [&](){return normal(eng);});

    naft::Document doc(std::cout);

    double ll = 0.0;
    Float step = 0.00015;
    for (int i = 0; i < 200; ++i)
    {
        double newll;
        REQUIRE(trainer.train(newll, weights.data(), 0.1, 10.0, step));
        step *= (newll < ll ? 0.5 : 1.01);
        ll = newll;
        doc.node("train").attr("ll",ll).attr("step",step);
    }

    if (true)
    {
        Vector states(nn.nr_states());
        states[bias] = 1.0;
        for (Float x = -2.0; x <= 2.0; x += 0.01)
        {
            states[input] = x;
            nn.forward(states.data(), weights.data());
            const auto o = states[output];
            doc.node("point").attr("x",x).attr("t",std::sin(x)).attr("o",o);
        }
    }
}
