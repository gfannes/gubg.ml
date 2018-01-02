#include "catch.hpp"
#include "gubg/ml/MLP.hpp"
#include "gubg/ml/Cost.hpp"
#include "gubg/optimization/SCG.hpp"
#include <cmath>
using namespace gubg;

namespace  { 
    using Float = double;
    using Params = ml::mlp::Params<Float>;
    using Model = ml::mlp::Model<Float>;
    using Input = Model::Input;
    using Output = Model::Output;

    struct ParamsInfo
    {
        using Type = Params;

        static void fill(Type &params, Float v)
        {
            for (auto &layer: params.layers)
            {
                layer.weights.fill(v);
                std::fill(RANGE(layer.biases), v);
            }
        }
        static unsigned int order(const Type &params)
        {
            unsigned int nr = 0;
            for (const auto &layer: params.layers)
                nr += layer.weights.nr_rows()*layer.weights.nr_cols() + layer.biases.size();
            return nr;
        }
        static Float sum_squares(const Type &params)
        {
            Float ss = 0.0;
            for (const auto &layer: params.layers)
            {
                ss += layer.weights.sum_squares();
                for (const auto v: layer.biases)
                    ss += v*v;
            }
            return ss;
        }
        static Float inprod(const Type &a, const Type &b)
        {
            Float res = 0.0;
            const auto nr_layers = a.layers.size();
            assert(b.layers.size() == nr_layers);
            for (size_t lix = 0; lix < nr_layers; ++lix)
            {
                const auto &la = a.layers[lix];
                const auto &lb = b.layers[lix];
                {
                    Float tmp;
                    la.weights.inproduct(tmp, lb.weights);
                    res += tmp;
                }
                const auto nr_biases = la.biases.size();
                for (size_t bix = 0; bix < nr_biases; ++bix)
                    res += la.biases[bix]*lb.biases[bix];
            }
            return res;
        }
        static void update(Type &dst, Float factor, const Type &src)
        {
            const auto nr_layers = src.layers.size();
            assert(dst.layers.size() == nr_layers);
            for (size_t lix = 0; lix < nr_layers; ++lix)
            {
                const auto &src_layer = src.layers[lix];
                auto &dst_layer = dst.layers[lix];
                dst_layer.weights.add(src_layer.weights, factor);
                auto it = dst_layer.biases.begin();
                for (auto v: src_layer.biases)
                    *it++ += factor*v;
            }
        }
    };

    struct Outer
    {
        void scg_params(unsigned int iter, Float cost, const Params &params)
        {
        }
        bool scg_terminate(unsigned int iter, Float cost, const Params &gradient)
        {
            return iter > 10;
        }
    };
} 

TEST_CASE("gubg::ml::MLP tests", "[ut][ml][MLP]")
{
    Model model;

    Params params(1, {10,10}, 1);

    Input input = {1.0};
    Output output = {999.0};

    const Tanh<Float> tanh;

    SECTION("default output should be 0")
    {
        REQUIRE(model(output, input, params));
        REQUIRE(output[0] == tanh(0.0));
    }
    SECTION("when the output bias is set, this should affect the output")
    {
        params.layers[2].biases[0] = 1.0;
        REQUIRE(model(output, input, params));
        REQUIRE(output[0] == tanh(1.0));
    }

    SECTION("learn a sine wave")
    {
        struct MeanModel
        {
            const Params *params = nullptr;
            Model model;
            bool operator()(Output &output, const Input &input) const
            {
                return model(output, input, *params);
            }
        };
        struct CostFunction
        {
            Float sigma = 1.0;

            bool operator()(Float &cost, const Output &mean, const Output &actual) const
            {
                MSS_BEGIN(bool);
                MSS(mean.size() == actual.size());
                cost = 0.0;
                const auto size = mean.size();
                for (size_t ix = 0; ix < size; ++ix)
                {
                    const auto diff = (mean[ix]-actual[ix]);
                    cost += diff*diff/sigma;
                }
                MSS_END();
            }
        };
        ml::Cost<Input, Output, MeanModel, CostFunction> cost;
        cost.mean_model.params = &params;
        for (double t = 0.0; t < 12.0; t += 0.1)
        {
            cost.inputs.push_back({t});
            cost.outputs.push_back({0.8*std::sin(t)});
        }
        double c;
        REQUIRE(cost(c));
        std::cout << C(c) << std::endl;

        Outer outer;
        optimization::SCG<Float, ParamsInfo, Outer> scg(outer);

        auto function = [&](const Params &params)
        {
            Float c;
            cost.mean_model.params = &params;
            cost(c);
            return c;
        };
        auto gradient = [&](Params &grad, const Params &params)
        {
            S("");
            cost.mean_model.params = &params;
            ParamsInfo::fill(grad, 0.0);
            Params tmp = grad;
            assert(cost.inputs.size() == cost.outputs.size());
            const auto nr = cost.inputs.size();
            assert(nr > 0);
            Output mean = cost.outputs[0];
            for (size_t i = 0; i < nr; ++i)
            {
                cost.mean_model.model.gradient(tmp, mean, cost.inputs[i], *cost.mean_model.params);
                ParamsInfo::update(grad, 2.0*(mean[0]-cost.outputs[i][0]), tmp);
            }
        };
        const auto new_c = scg(params, function, gradient);
        std::cout << C(new_c) << std::endl;
    }
}
