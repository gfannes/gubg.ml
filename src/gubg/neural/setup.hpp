#ifndef HEADER_gubg_neural_setup_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_setup_hpp_ALREADY_INCLUDED

#include "gubg/neural/Simulator.hpp"
#include "gubg/mlp/Structure.hpp"
#include "gubg/mss.hpp"
#include <vector>
#include <numeric>

namespace gubg { namespace neural { 

    template <typename Simulator, typename IX>
    bool setup(Simulator &simulator, const mlp::Structure &structure, IX &first_input, IX &bias, IX &first_output)
    {
        MSS_BEGIN(bool);

        simulator.clear();

        auto nr_inputs = structure.nr_inputs;
        first_input = simulator.add_external(nr_inputs);
        bias = simulator.add_external(1);

        std::vector<size_t> inputs;
        size_t local_first_input = first_input;
        for (const auto &layer: structure.layers)
        {
            inputs.resize(nr_inputs);
            std::iota(RANGE(inputs), local_first_input);
            inputs.push_back(bias);
            MSS(layer.neurons.size() > 0);
            for (auto nix = 0u; nix < layer.neurons.size(); ++nix)
            {
                const auto &neuron = layer.neurons[nix];
                if (nix == 0)
                    //We treat the first neuron special: we need to know where it puts its output.
                    //All the others are just after this first neuron.
                    MSS(simulator.add_neuron(neuron.transfer, inputs, local_first_input));
                else
                    MSS(simulator.add_neuron(neuron.transfer, inputs));
            }
            nr_inputs = layer.neurons.size();
        }
        first_output = local_first_input;

        MSS_END();
    }

} } 

#endif
