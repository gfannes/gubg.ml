#include <gubg/ann/Transfer.hpp>

namespace gubg { namespace ann {

    const char *to_str(Transfer transfer)
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

    bool from_str(Transfer &transfer, const std::string &str)
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

} }
