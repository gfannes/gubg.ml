#ifndef HEADER_gubg_mlp_Parameters_hpp_ALREADY_INCLUDED
#define HEADER_gubg_mlp_Parameters_hpp_ALREADY_INCLUDED

#include "gubg/mlp/Structure.hpp"
#include "gubg/mss.hpp"
#include <vector>

namespace gubg { namespace mlp { 

    struct Parameters
    {
        struct Layer
        {
            struct Neuron
            {
                double bias = 0.0;
                std::vector<double> weights;

                template <typename Writer> bool write(Writer &w) const;
                template <typename Reader> bool read(Reader &r, unsigned int nr_inputs);
            };
            std::vector<Neuron> neurons;

            template <typename Writer> bool write(Writer &w) const;
            template <typename Reader> bool read(Reader &r, unsigned int nr_inputs);
        };
        unsigned int nr_inputs = 0;
        std::vector<Layer> layers;

        void setup(const Structure &s);

        template <typename Writer> bool write(Writer &w) const;
        template <typename Reader> bool read(Reader &r);
    };

    //Parameters
    inline void Parameters::setup(const Structure &s)
    {
        nr_inputs = s.nr_inputs;
        auto local_nr_inputs = nr_inputs;
        layers.resize(s.layers.size());
        for (auto lix = 0u; lix < layers.size(); ++lix)
        {
            auto &layer = layers[lix];
            const auto nr_neurons = s.layers[lix].neurons.size();
            layer.neurons.resize(nr_neurons);
            for (auto &neuron: layer.neurons)
                neuron.weights.resize(local_nr_inputs);
            local_nr_inputs = nr_neurons;
        }
    }
    template <typename Writer>
    bool Parameters::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        MSS(w.attr("nr_inputs", nr_inputs));
        MSS(w.attr("nr_layers", layers.size()));
        for (auto lix = 0u; lix < layers.size(); ++lix) { MSS(w.object(lix, layers[lix])); }
        MSS_END();
    }
    template <typename Reader>
    bool Parameters::read(Reader &r)
    {
        MSS_BEGIN(bool);
        MSS(r.attr("nr_inputs", nr_inputs));
        {
            unsigned int nr_layers;
            MSS(r.attr("nr_layers", nr_layers));
            layers.resize(nr_layers);
        }
        auto local_nr_inputs = nr_inputs;
        for (auto lix = 0u; lix < layers.size(); ++lix)
        {
            auto &layer = layers[lix];
            MSS(r.object(lix, layer, local_nr_inputs));
            local_nr_inputs = layer.neurons.size();
        }
        MSS_END();
    }

    //Parameters::Layer
    template <typename Writer>
    bool Parameters::Layer::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        MSS(w.attr("nr_neurons", neurons.size()));
        for (auto nix = 0u; nix < neurons.size(); ++nix) { MSS(w.object(nix, neurons[nix])); }
        MSS_END();
    }
    template <typename Reader>
    bool Parameters::Layer::read(Reader &r, unsigned int nr_inputs)
    {
        MSS_BEGIN(bool);
        {
            unsigned int nr_neurons;
            MSS(r.attr("nr_neurons", nr_neurons));
            neurons.resize(nr_neurons);
        }
        for (auto nix = 0u; nix < neurons.size(); ++nix)
        {
            MSS(r.object(nix, neurons[nix], nr_inputs));
        }
        MSS_END();
    }

    //Parameters::Layer::Neuron
    template <typename Writer>
    bool Parameters::Layer::Neuron::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        MSS(w.attr("bias", bias));
        for (auto wix = 0u; wix < weights.size(); ++wix) { MSS(w.attr(wix, weights[wix])); }
        MSS_END();
    }
    template <typename Reader>
    bool Parameters::Layer::Neuron::read(Reader &r, unsigned int nr_inputs)
    {
        MSS_BEGIN(bool);
        MSS(r.attr("bias", bias));
        weights.resize(nr_inputs);
        {
            bool ok = true;
            auto set_weight = [&](const std::string &key, const std::string &value){
                const auto wix = std::stoul(key);
                AGG(ok, wix < weights.size());
                weights[wix] = std::stod(value);
            };
            MSS(ok);
        }
        MSS_END();
    }

} } 

#endif
