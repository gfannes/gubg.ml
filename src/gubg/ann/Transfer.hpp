#ifndef HEADER_gubg_ann_Types_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Types_hpp_ALREADY_INCLUDED

#include <gubg/Tanh.hpp>
#include <gubg/Sigmoid.hpp>
#include <string>

namespace gubg { namespace ann { 

    enum class Transfer
    {
        Linear, Tanh, Sigmoid, LeakyReLU, SoftPlus, Quadratic,
    };

    inline const char *to_str(Transfer transfer)
    {
        switch (transfer)
        {
#define L_CASE(name) case Transfer::name: return #name
            L_CASE(Linear);
            L_CASE(Tanh);
            L_CASE(Sigmoid);
            L_CASE(LeakyReLU);
            L_CASE(SoftPlus);
            L_CASE(Quadratic);
#undef L_CASE
            default: break;
        }
        return "Unknown Transfer";
    }

    inline bool from_str(Transfer &transfer, const std::string &str)
    {
#define L_IF(name) if (str == #name) {transfer = Transfer::name; return true;}
        L_IF(Linear)
        L_IF(Tanh)
        L_IF(Sigmoid)
        L_IF(LeakyReLU)
        L_IF(SoftPlus)
        L_IF(Quadratic)
#undef L_ELSE_IF

        return false;
    }

    namespace transfer { 
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
    } 

} } 

#endif
