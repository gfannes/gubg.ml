#ifndef HEADER_gubg_neural_Network_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_Network_hpp_ALREADY_INCLUDED

#include "gubg/Tanh.hpp"
#include "gubg/Sigmoid.hpp"
#include "gubg/Range.hpp"
#include "gubg/mss.hpp"
#include <vector>
#include <list>
#include <memory>
#include <cassert>

namespace gubg { namespace neural { 

    enum class Transfer
    {
        Linear, Tanh, Sigmoid, LeakyReLU, 
    };

    namespace transfer { 
        struct Identity
        {
            template <typename Float>
            void operator()(Float &) const {}
        };
        struct Tanh
        {
            template <typename Float>
            void operator()(Float &v) const
            {
                gubg::Tanh<Float> tanh;
                v = tanh(v);
            }
        };
        struct Sigmoid
        {
            template <typename Float>
            void operator()(Float &v) const
            {
                gubg::Sigmoid<Float> sigmoid;
                v = sigmoid(v);
            }
        };
        struct LeakyReLU
        {
            template <typename Float>
            void operator()(Float &v) const {if (v < 0) v *= 0.01;}
        };
    } 

    namespace neuron { 
        template <typename Float>
        class Interface
        {
        private:
            using Self = Interface<Float>;
        public:
            using Ptr = std::unique_ptr<Self>;
            using Inputs = std::vector<size_t>;
            using Output = size_t;
            using Weight = size_t;

            Inputs inputs;
            Output output;
            Weight weight;

            virtual ~Interface() {}

            virtual Float *forward(Float *states, const Float *weights) const = 0;
            virtual Float *forward(Float *states, Float *preacts, const Float *weights) const = 0;

        protected:
            Float &preactivate_(Float *states, const Float *weights) const
            {
                auto &dst = states[output];
                dst = 0.0;
                auto w = weights+weight;
                for (auto src: inputs)
                    dst += (*w++)*states[src];
                return dst;
            }
        };

        template <typename Float, typename Transfer>
        class Neuron: public Interface<Float>
        {
        private:
            using Base = Interface<Float>;
        public:
            Float *forward(Float *states, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(states, weights);
                transfer_(dst);
                return &dst;
            }
            Float *forward(Float *states, Float *preacts, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(states, weights);
                preacts[Base::output] = dst;
                transfer_(dst);
                return &dst;
            }
        private:
            Transfer transfer_;
        };


        template <typename Float>
        Interface<Float> *create(Transfer tf)
        {
            switch (tf)
            {
                case Transfer::Linear:    return new Neuron<Float, transfer::Identity>;
                case Transfer::Tanh:      return new Neuron<Float, transfer::Tanh>;
                case Transfer::Sigmoid:   return new Neuron<Float, transfer::Sigmoid>;
                case Transfer::LeakyReLU: return new Neuron<Float, transfer::LeakyReLU>;
            }
            assert(false);
            return nullptr;
        }
    } 

    template <typename Float>
    class Network
    {
    public:
        size_t nr_states() const {return nr_states_;}
        size_t nr_weights() const {return nr_weights_;}

        size_t add_external(size_t size)
        {
            const auto ix = nr_states_;
            nr_states_ += size;
            return ix;
        }

        //Optional arguments output and weight will provide info on where the neuron
        //will store its output in the states array, and where the first weight starts
        template <typename Inputs>
        bool add_neuron(Transfer tf, const Inputs &inputs) { return add_neuron_<Inputs, size_t>(tf, inputs, nullptr, nullptr); }
        template <typename Inputs, typename IX>
        bool add_neuron(Transfer tf, const Inputs &inputs, IX &output) { return add_neuron_<Inputs, IX>(tf, inputs, &output, nullptr); }
        template <typename Inputs, typename IX>
        bool add_neuron(Transfer tf, const Inputs &inputs, IX &output, IX &weight) { return add_neuron_<Inputs, IX>(tf, inputs, &output, &weight); }

        //Expects input at start of states
        void forward(Float *states, const Float *weights) const
        {
            for (const auto &neuron: neurons_)
                neuron->forward(states, weights);
        }
        void forward(Float *states, Float *preacts, const Float *weights) const
        {
            for (const auto &neuron: neurons_)
                neuron->forward(states, preacts, weights);
        }

    private:
        using Neuron_itf = neuron::Interface<Float>;
        using Neurons = std::list<typename Neuron_itf::Ptr>;

        template <typename Inputs, typename IX>
        bool add_neuron_(Transfer tf, const Inputs &inputs, IX *output, IX *weight)
        {
            MSS_BEGIN(bool);
            neurons_.emplace_back(neuron::create<Float>(tf));
            auto &ptr = neurons_.back();
            MSS(!!ptr, neurons_.pop_back());
            auto &n = *ptr;
            n.inputs.assign(RANGE(inputs));
            n.output = nr_states_++;
            if (output)
                *output = n.output;
            n.weight = nr_weights_;
            if (weight)
                *weight = n.weight;
            nr_weights_ += inputs.size();
            MSS_END();
        }

        size_t nr_states_ = 0;
        size_t nr_weights_ = 0;

        Neurons neurons_;
    };

} } 

#endif
