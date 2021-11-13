#ifndef HEADER_gubg_ann_Transfer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Transfer_hpp_ALREADY_INCLUDED

#include <gubg/Tanh.hpp>
#include <gubg/Sigmoid.hpp>
#include <string>
#include <cassert>

namespace gubg { namespace ann {

    enum class Transfer
    {
        Linear, Tanh, Sigmoid, LeakyReLU, SoftPlus, Quadratic,
    };

    const char *to_str(Transfer transfer);
    bool from_str(Transfer &transfer, const std::string &str);

    namespace transfer { 

        template <typename T>
        T output(T v, Transfer transfer);

        struct Linear
        {
            template <typename Float>
            static Float output(Float x) {return x;}
            template <typename Float>
            static Float derivative(Float x) {return 1.0;}
        };
        struct Tanh
        {
            template <typename Float>
            static Float output(Float x)
            {
                gubg::Tanh<Float> tanh;
                return tanh(x);
            }
            template <typename Float>
            static Float derivative(Float x)
            {
                x = output(x);
                return 1.0-x*x;
            }
        };
        struct Sigmoid
        {
            template <typename Float>
            static Float output(Float x)
            {
                gubg::Sigmoid<Float> sigmoid;
                return sigmoid(x);
            }
            template <typename Float>
            static Float derivative(Float x)
            {
                x = output(x);
                return x*(1.0-x);
            }
        };
        struct LeakyReLU
        {
            template <typename Float>
            static Float output(Float x) {return (x >= 0 ? x : x*0.01);}
            template <typename Float>
            static Float derivative(Float x)
            {
                if (x < 0)
                    return 0.01;
                return 1.0;
            }
        };
        struct SoftPlus
        {
            template <typename Float>
            static Float output(Float x) {return std::log(1.0+std::exp(x));}
            template <typename Float>
            static Float derivative(Float x) { return 1.0/(1.0+std::exp(-x)); }
        };
        struct Quadratic
        {
            template <typename Float>
            static Float output(Float x) {return x*x*0.5;}
            template <typename Float>
            static Float derivative(Float x)
            {
                return x;
            }
        };
    
        template <typename T>
        T output(T v, Transfer transfer)
        {
            switch (transfer)
            {
                case Transfer::Linear:    v = Linear::output(v); break;
                case Transfer::Tanh:      v = Tanh::output(v); break;
                case Transfer::Sigmoid:   v = Sigmoid::output(v); break;
                case Transfer::LeakyReLU: v = LeakyReLU::output(v); break;
                case Transfer::SoftPlus:  v = SoftPlus::output(v); break;
                case Transfer::Quadratic: v = Quadratic::output(v); break;
                default: assert(false);   v = 0; break;
            }
            return v;
        }
    
        template <typename T>
        T derivative(T v, Transfer transfer)
        {
            switch (transfer)
            {
                case Transfer::Linear:    v = Linear::derivative(v); break;
                case Transfer::Tanh:      v = Tanh::derivative(v); break;
                case Transfer::Sigmoid:   v = Sigmoid::derivative(v); break;
                case Transfer::LeakyReLU: v = LeakyReLU::derivative(v); break;
                case Transfer::SoftPlus:  v = SoftPlus::derivative(v); break;
                case Transfer::Quadratic: v = Quadratic::derivative(v); break;
                default: assert(false);   v = 0; break;
            }
            return v;
        }
    }
} } 

#endif
