#ifndef HEADER_gubg_mlp_Structure_hpp_ALREADY_INCLUDED
#define HEADER_gubg_mlp_Structure_hpp_ALREADY_INCLUDED

#include <gubg/ann/Transfer.hpp>
#include <gubg/mss.hpp>
#include <vector>

namespace gubg { namespace mlp { 

    struct Structure
    {
        unsigned int nr_inputs = 0;

        struct Layer
        {
            struct Neuron
            {
                ann::Transfer transfer = ann::Transfer::Linear;
                double weight_stddev = 1.0;
                double bias_stddev = 1.0;

                template <typename Writer> bool write(Writer &w) const;
                template <typename Reader> bool read(Reader &r);
            };
            std::vector<Neuron> neurons;

            template <typename Ftor> void add_neuron(Ftor &&ftor);

            template <typename Writer> bool write(Writer &w) const;
            template <typename Reader> bool read(Reader &r);
        };
        std::vector<Layer> layers;

        Structure() {}
        Structure(unsigned int nr_inputs): nr_inputs(nr_inputs) {}

        unsigned int nr_outputs() const {return layers.back().neurons.size();}

        const Layer::Neuron &neuron(size_t lix, size_t nix) const {return layers[lix].neurons[nix];}

        template <typename Ftor> void add_layer(Ftor &&ftor);
        void add_layer(ann::Transfer, unsigned int nr_neurons, double weight_stddev, double bias_stddev);

        template <typename Writer> bool write(Writer &w) const;
        template <typename Reader> bool read(Reader &r);
    };

    //Structure
    template <typename Ftor>
    void Structure::add_layer(Ftor &&ftor)
    {
        layers.emplace_back();
        ftor(layers.back());
    }
    inline void Structure::add_layer(ann::Transfer transfer, unsigned int nr_neurons, double weight_stddev, double bias_stddev)
    {
        add_layer([&](auto &layer){
                for (auto i = 0u; i < nr_neurons; ++i)
                layer.add_neuron([&](auto &neuron){
                        neuron.transfer = transfer;
                        neuron.weight_stddev = weight_stddev;
                        neuron.bias_stddev = bias_stddev;
                        });
                });
    }
    template <typename Writer>
    bool Structure::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        MSS(w.attr("nr_inputs", nr_inputs));
        for (const auto &layer: layers) { MSS(w.object("layer", layer)); }
        MSS_END();
    }
    template <typename Reader>
    bool Structure::read(Reader &r)
    {
        MSS_BEGIN(bool);

        MSS(r.attr("nr_inputs", nr_inputs));

        auto read_layer = [&](Reader &r){
            layers.emplace_back();
            return layers.back().read(r);
        };
        layers.clear();
        while (r("layer", read_layer)) {}

        MSS_END();
    }

    //Structure::Layer
    template <typename Ftor>
    void Structure::Layer::add_neuron(Ftor &&ftor)
    {
        neurons.emplace_back();
        ftor(neurons.back());
    }
    template <typename Writer>
    bool Structure::Layer::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        for (const auto &neuron: neurons) { MSS(w.object("neuron", neuron)); }
        MSS_END();
    }
    template <typename Reader>
    bool Structure::Layer::read(Reader &r)
    {
        MSS_BEGIN(bool);
        auto read_neuron = [&](Reader &r){
            neurons.emplace_back();
            return neurons.back().read(r);
        };
        while (r("neuron", read_neuron)) {}
        MSS_END();
    }

    //Structure::Layer::Neuron
    template <typename Writer>
    bool Structure::Layer::Neuron::write(Writer &w) const
    {
        MSS_BEGIN(bool);
        MSS(w.attr_ftor("transfer", [&]{ return to_str(transfer); }));
        MSS(w.attr("weight.stddev", weight_stddev));
        MSS(w.attr("bias.stddev", bias_stddev));
        MSS_END();
    }
    template <typename Reader>
    bool Structure::Layer::Neuron::read(Reader &r)
    {
        MSS_BEGIN(bool);
        MSS(r.attr_ftor("transfer", [&](const std::string &str){ return from_str(transfer, str); }));
        MSS(r.attr("weight.stddev", weight_stddev));
        MSS(r.attr("bias.stddev", bias_stddev));
        MSS_END();
    }

} } 

#endif
