#ifndef HEADER_gubg_neural_setup_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_setup_hpp_ALREADY_INCLUDED

#include "gubg/neural/Simulator.hpp"
#include "gubg/mlp/Structure.hpp"
#include "gubg/mlp/Parameters.hpp"
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
        //Bias comes after the inputs
        bias = simulator.add_external(1);

        std::vector<size_t> inputs;
        size_t local_first_input = first_input;
        for (const auto &layer: structure.layers)
        {
            inputs.resize(nr_inputs);
            std::iota(RANGE(inputs), local_first_input);
            //Bias comes after the inputs
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

    template <typename Weights>
    bool setup(Weights &weights, const mlp::Parameters &params)
    {
        MSS_BEGIN(bool);

        auto nr_left = weights.size();
        auto ptr = weights.data();

        //Bias comes after the inputs
        auto bias = params.nr_inputs;

        auto local_nr_inputs = params.nr_inputs;
        for (const auto &layer: params.layers)
        {
            for (const auto &neuron: layer.neurons)
            {
                const auto nr_weights = neuron.weights.size();

                MSS(nr_weights == local_nr_inputs);
                MSS(nr_weights+1 <= nr_left);

                std::copy(RANGE(neuron.weights), ptr); ptr += nr_weights;
                *ptr++ = neuron.bias;
                nr_left -= nr_weights+1;
            }
            local_nr_inputs = layer.neurons.size();
        }
        MSS(nr_left == 0);

        MSS_END();
    }

} } 

#endif
