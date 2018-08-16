#ifndef HEADER_gubg_neural_Types_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_Types_hpp_ALREADY_INCLUDED

#include "gubg/mss.hpp"
#include <string>

namespace gubg { namespace neural { 

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
        MSS_BEGIN(bool);
        if (false) {}
#define L_ELSE_IF(name) else if (str == #name) {transfer = Transfer::name;}
        L_ELSE_IF(Linear)
        L_ELSE_IF(Tanh)
        L_ELSE_IF(Sigmoid)
        L_ELSE_IF(LeakyReLU)
        L_ELSE_IF(SoftPlus)
        L_ELSE_IF(Quadratic)
#undef L_ELSE_IF
        else {MSS(false);}
        MSS_END();
    }

} } 

#endif
