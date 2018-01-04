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

            virtual Float *forward(Float *postacts, const Float *weights) const = 0;

        protected:
            Float &preactivate_(Float *postacts, const Float *weights) const
            {
                auto &dst = postacts[output];
                dst = 0.0;
                auto w = weights+weight;
                for (auto src: inputs)
                    dst += (*w++)*postacts[src];
                return dst;
            }
        };

        template <typename Float>
        class Linear: public Interface<Float>
        {
        private:
            using Base = Interface<Float>;
        public:
            Float *forward(Float *postacts, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(postacts, weights);
                return &dst;
            }
        private:
        };

        template <typename Float>
        class Tanh: public Interface<Float>
        {
        private:
            using Base = Interface<Float>;
        public:
            Float *forward(Float *postacts, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(postacts, weights);
                dst = tanh_(dst);
                return &dst;
            }
        private:
            gubg::Tanh<Float> tanh_;
        };

        template <typename Float>
        class Sigmoid: public Interface<Float>
        {
        private:
            using Base = Interface<Float>;
        public:
            Float *forward(Float *postacts, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(postacts, weights);
                dst = sigmoid_(dst);
                return &dst;
            }
        private:
            gubg::Sigmoid<Float> sigmoid_;
        };

        template <typename Float>
        class LeakyReLU: public Interface<Float>
        {
        private:
            using Base = Interface<Float>;
        public:
            Float *forward(Float *postacts, const Float *weights) const override
            {
                auto &dst = Base::preactivate_(postacts, weights);
                if (dst < 0)
                    dst *= 0.01;
                return &dst;
            }
        private:
        };

        template <typename Float>
        Interface<Float> *create(Transfer transfer)
        {
            switch (transfer)
            {
                case Transfer::Linear:    return new Linear<Float>;
                case Transfer::Tanh:      return new Tanh<Float>;
                case Transfer::Sigmoid:   return new Sigmoid<Float>;
                case Transfer::LeakyReLU: return new LeakyReLU<Float>;
            }
            assert(false);
            return nullptr;
        }
    } 

    template <typename Float>
    class Network
    {
    public:
        Network(size_t nr_inputs): nr_inputs_(nr_inputs)
        {
            nr_postacts_ = nr_inputs_;
            nr_postacts_ += 1;
        }

        size_t nr_weights() const {return nr_weights_;}
        size_t nr_postacts() const {return nr_postacts_;}

        size_t input(size_t offset) const {return 0+offset;}
        size_t bias() const {return nr_inputs_;}

        //Optional arguments output and weight will provide info on where the neuron
        //will store its output in the postacts array, and where the first weight starts
        template <typename Inputs, typename IX>
        bool add_neuron(Transfer transfer, const Inputs &inputs, IX *output = nullptr, IX *weight = nullptr)
        {
            MSS_BEGIN(bool);
            neurons_.emplace_back(neuron::create<Float>(transfer));
            auto &ptr = neurons_.back();
            MSS(!!ptr, neurons_.pop_back());
            auto &n = *ptr;
            n.inputs.assign(RANGE(inputs));
            n.output = nr_postacts_++;
            if (output)
                *output = n.output;
            n.weight = nr_weights_;
            if (weight)
                *weight = n.weight;
            nr_weights_ += inputs.size();
            MSS_END();
        }

        //Expects input at start of postacts
        void forward(Float *postacts, const Float *weights) const
        {
            postacts[bias()] = 1.0;
            for (const auto &neuron: neurons_)
            {
                neuron->forward(postacts, weights);
            }
        }

    private:
        using Neuron_itf = neuron::Interface<Float>;
        using Neurons = std::list<typename Neuron_itf::Ptr>;

        const size_t nr_inputs_;
        size_t nr_weights_ = 0;
        size_t nr_postacts_;

        Neurons neurons_;
    };

} } 

#endif
