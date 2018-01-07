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

    namespace unit { 
        template <typename Float>
        class Interface
        {
        private:
            using Self = Interface<Float>;
        public:
            using Ptr = std::unique_ptr<Self>;

            virtual ~Interface() {}

            virtual void forward(Float *states, const Float *weights) const = 0;
            virtual void forward(Float *states, Float *preacts, const Float *weights) const = 0;

            //derivative: per state
            //gradient: per weight
            virtual void backward(Float *derivative, Float *gradient, const Float *states, const Float *preacts, const Float *weights) const = 0;
        };
    } 

    enum class Transfer
    {
        Linear, Tanh, Sigmoid, LeakyReLU, Quadratic,
    };

    namespace transfer { 
        struct Identity
        {
            template <typename Float>
            void operator()(Float &) const {}
            template <typename Float>
            Float derivative(Float x) const {return 1.0;}
        };
        struct Tanh
        {
            template <typename Float>
            void operator()(Float &v) const
            {
                gubg::Tanh<Float> tanh;
                v = tanh(v);
            }
            template <typename Float>
            Float derivative(Float x) const
            {
                operator()(x);
                return 1.0-x*x;
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
            template <typename Float>
            Float derivative(Float x) const
            {
                operator()(x);
                return x*(1.0-x);
            }
        };
        struct LeakyReLU
        {
            template <typename Float>
            void operator()(Float &v) const {if (v < 0) v *= 0.01;}
            template <typename Float>
            Float derivative(Float x) const
            {
                if (x < 0)
                    return 0.01;
                return 1.0;
            }
        };
        struct Quadratic
        {
            template <typename Float>
            void operator()(Float &v) const {v = v*v*0.5;}
            template <typename Float>
            Float derivative(Float x) const
            {
                return x;
            }
        };
    } 

    namespace cost { 
        template <typename Float>
        class Quadratic: public unit::Interface<Float>
        {
        public:
            Quadratic(size_t mean, size_t wanted, size_t output): mean_(mean), wanted_(wanted), output_(output) { }

            void forward(Float *states, const Float *weights) const override
            {
                states[output_] = (states[mean_] - states[wanted_]);
                quadratic_(states[output_]);
            }
            void forward(Float *states, Float *preacts, const Float *weights) const override
            {
                preacts[output_] = states[output_] = (states[mean_] - states[wanted_]);
                quadratic_(states[output_]);
            }

            void backward(Float *derivative, Float *gradient, const Float *states, const Float *preacts, const Float *weights) const override
            {
                auto &src = derivative[output_];
                src *= quadratic_.derivative(preacts[output_]);
                derivative[mean_] += src;
                derivative[wanted_] -= src;
            }
        private:
            const size_t mean_;
            const size_t wanted_;
            const size_t output_;
            transfer::Quadratic quadratic_;
        };

        template <typename Float>
        class Sum: public unit::Interface<Float>
        {
        public:
            using IXs = std::vector<size_t>;

            template <typename IXs>
            Sum(const IXs &ixs, size_t output, Float factor = 1.0): output_(output), factor_(factor), inputs_(RANGE(ixs)) { }

            void forward(Float *states, const Float *weights) const override
            {
                Float sum = 0.0;
                for (auto input: inputs_)
                    sum += states[input];
                states[output_] = sum*factor_;
            }
            void forward(Float *states, Float *preacts, const Float *weights) const override
            {
                forward(states, weights);
                preacts[output_] = states[output_];
            }

            void backward(Float *derivative, Float *gradient, const Float *states, const Float *preacts, const Float *weights) const override
            {
                auto &src = derivative[output_];
                src *= factor_;
                for (auto input: inputs_)
                    derivative[input] += src;
            }

        private:
            const size_t output_;
            const Float factor_;
            IXs inputs_;
        };
    } 

    namespace neuron { 
        template <typename Float, typename Transfer>
        class Neuron: public unit::Interface<Float>
        {
        private:
            using Self = unit::Interface<Float>;

        public:
            using Ptr = std::unique_ptr<Self>;
            using Inputs = std::vector<size_t>;
            using Output = size_t;
            using Weight = size_t;

            template <typename Inputs>
            Neuron(const Inputs &inputs, size_t output, size_t weight): inputs_(RANGE(inputs)), output_(output), weight_(weight) {}

            void forward(Float *states, const Float *weights) const override
            {
                auto &dst = preactivate_(states, weights);
                transfer_(dst);
            }
            void forward(Float *states, Float *preacts, const Float *weights) const override
            {
                auto &dst = preactivate_(states, weights);
                preacts[output_] = dst;
                transfer_(dst);
            }

            void backward(Float *derivative, Float *gradient, const Float *states, const Float *preacts, const Float *weights) const override
            {
                auto &src = derivative[output_];
                src *= transfer_.derivative(preacts[output_]);
                auto w = weights+weight_;
                auto g = gradient+weight_;
                for (auto ix: inputs_)
                {
                    *g++ += src*states[ix];
                    derivative[ix] += src*(*w++);
                }
            }
        private:
            Float &preactivate_(Float *states, const Float *weights) const
            {
                auto &dst = states[output_];
                dst = 0.0;
                auto w = weights+weight_;
                for (auto src: inputs_)
                    dst += (*w++)*states[src];
                return dst;
            }

            Inputs inputs_;
            Output output_;
            Weight weight_;
            Transfer transfer_;
        };

        template <typename Float, typename Inputs>
        unit::Interface<Float> *create(Transfer tf, const Inputs &inputs, size_t output, size_t weight)
        {
            switch (tf)
            {
                case Transfer::Linear:    return new Neuron<Float, transfer::Identity>(inputs, output, weight);
                case Transfer::Tanh:      return new Neuron<Float, transfer::Tanh>(inputs, output, weight);
                case Transfer::Sigmoid:   return new Neuron<Float, transfer::Sigmoid>(inputs, output, weight);
                case Transfer::LeakyReLU: return new Neuron<Float, transfer::LeakyReLU>(inputs, output, weight);
                case Transfer::Quadratic: return new Neuron<Float, transfer::Quadratic>(inputs, output, weight);
            }
            return nullptr;
        }
    } 

    template <typename Float>
    class Network
    {
    public:
        size_t nr_states() const {return nr_states_;}
        size_t nr_weights() const {return nr_weights_;}

        //Adds <size> states that should be given values externally, e.g.,
        //inputs, bias input or target outputs
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

        template <typename IXs, typename IX>
        bool add_loglikelihood(const IXs &mean, const IXs &wanted, IX &output, Float stddev = 1.0)
        {
            MSS_BEGIN(bool);
            MSS(mean.size() == wanted.size());
            const auto size = mean.size();
            std::vector<size_t> quads(size);
            for (size_t i = 0; i < size; ++i)
            {
                const size_t out = nr_states_++;
                units_.emplace_back(new cost::Quadratic<Float>(mean[i], wanted[i], out));
                quads[i] = out;
            }
            output = nr_states_++;
            units_.emplace_back(new cost::Sum<Float>(quads, output, 1.0/stddev/stddev));
            MSS_END();
        }

        void forward(Float *states, const Float *weights) const
        {
            for (const auto &neuron: units_)
                neuron->forward(states, weights);
        }
        void forward(Float *states, Float *preacts, const Float *weights) const
        {
            for (const auto &neuron: units_)
                neuron->forward(states, preacts, weights);
        }

        //Caller has to setup derivative each time: filled with zeros, the output should contain the cost
        //Gradient will be updated (+=), set it to 0 as appropriate
        void backward(Float *derivative, Float *gradient, const Float *states, const Float *preacts, const Float *weights)
        {
            const auto end = units_.rend();
            for (auto it = units_.rbegin(); it != end; ++it)
                (*it)->backward(derivative, gradient, states, preacts, weights);
        }

    private:
        using Unit_itf = unit::Interface<Float>;
        using Units = std::list<typename Unit_itf::Ptr>;

        template <typename Inputs, typename IX>
        bool add_neuron_(Transfer tf, const Inputs &inputs, IX *output, IX *weight)
        {
            MSS_BEGIN(bool);
            size_t o = nr_states_;
            size_t w = nr_weights_;
            auto ptr = neuron::create<Float>(tf, inputs, o, w);
            MSS(!!ptr);
            units_.emplace_back(ptr);
            ++nr_states_;
            nr_weights_ += inputs.size();
            if (output)
                *output = o;
            if (weight)
                *weight = w;
            MSS_END();
        }

        size_t nr_states_ = 0;
        size_t nr_weights_ = 0;

        Units units_;
    };

} } 

#endif
