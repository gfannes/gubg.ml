#ifndef HEADER_gubg_ml_MLP_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_MLP_hpp_ALREADY_INCLUDED

#include "gubg/Matrix.hpp"
#include "gubg/Tanh.hpp"
#include "gubg/mss.hpp"
#include <vector>
#include <cassert>

namespace gubg { namespace ml { 

    namespace layer { 
        template <typename Float>
        struct Params
        {
            Matrix<Float> weights;
            std::vector<Float> biases;

            Params() {}
            Params(size_t nr_inputs, size_t nr_outputs): weights(nr_outputs, nr_inputs), biases(nr_outputs) {}

            size_t nr_outputs() const {return biases.size();}
        };

        template <typename Float>
        struct Model
        {
            template <typename Output, typename Input>
            bool operator()(Output &output, const Input &input, const Params<Float> &params) const
            {
                MSS_BEGIN(bool);
                MSS(params.weights.multiply(output, input));
                assert(output.size() == params.nr_outputs());
                const auto nr_output = output.size();
                Tanh<Float> tanh;
                for (size_t ix = 0; ix < nr_output; ++ix)
                    output[ix] = tanh(output[ix] + params.biases[ix]);
                MSS_END();
            }
        };
    } 

    namespace mlp { 
        template <typename Float>
        struct Params
        {
            using LayerParams = layer::Params<Float>;
            using Layers = std::vector<LayerParams>;

            Layers layers;

            Params() {}
            Params(size_t nr_inputs, const std::vector<size_t> &hidden_sizes, size_t nr_outputs): layers(hidden_sizes.size()+1)
            {
                auto layer = layers.begin();
                auto add_layer = [&](size_t nr_neurons)
                {
                    *layer++ = LayerParams(nr_inputs, nr_neurons);
                };

                for (auto nr_out: hidden_sizes)
                {
                    add_layer(nr_out);
                    nr_inputs = nr_out;
                }

                add_layer(nr_outputs);
            }
        };

        template <typename Float>
        class Model
        {
        public:
            using Vector = std::vector<Float>;
            using Input = Vector;
            using Output = Vector;

            bool operator()(Output &output, const Input &input, const Params<Float> &params) const
            {
                MSS_BEGIN(bool);
                tmp_outputs_.resize(params.layers.size());
                auto layer_params = params.layers.begin();
                const Input *inp = &input;
                Output *outp = nullptr;
                const layer::Model<Float> layer_model;
                for (auto &tmp_output: tmp_outputs_)
                {
                    tmp_output.resize(layer_params->nr_outputs());
                    MSS(layer_model(tmp_output, *inp, *layer_params++));
                    inp = &tmp_output;
                    outp = &tmp_output;
                }
                output.swap(*outp);
                MSS_END();
            }

            //You still need to take the cost function into account
            bool gradient(Params<Float> &gradient, Output &output, const Input &input, const Params<Float> &params) const
            {
                MSS_BEGIN(bool);
                MSS(operator()(output, input, params));
                MSS(gradient.layers.size() == params.layers.size());
                auto layer = gradient.layers.begin();
                const Input *inp = &input;
                for (const auto &tmp_output: tmp_outputs_)
                {
                    const size_t nro = tmp_output.size();
                    {
                        const auto &inpp = *inp;
                        const size_t nri = inpp.size();
                        auto &weights = layer->weights;
                        for (size_t o = 0; o < nro; ++o)
                            for (size_t i = 0; i < nri; ++i)
                                //TODO: This is hard-coded tanh since the derivative of tanh(x) => 1-tanh(x)*tanh(x)
                                weights.set(o, i, inpp[i]*(1.0-tmp_output[o]*tmp_output[o]));
                    }
                    {
                        auto &biases = layer->biases;
                        for (size_t o = 0; o < nro; ++o)
                            //TODO: This is hard-coded tanh since the derivative of tanh(x) => 1-tanh(x)*tanh(x)
                            biases[o] = 1.0-tmp_output[o]*tmp_output[o];
                    }
                    inp = &tmp_output;
                    ++layer;
                }
                MSS_END();
            }

        private:
            mutable std::vector<Vector> tmp_outputs_;
        };
    } 

} } 

#endif
