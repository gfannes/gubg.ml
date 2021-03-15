#ifndef HEADER_gubg_ml_cost_bitsize_Entropy_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_cost_bitsize_Entropy_hpp_ALREADY_INCLUDED

#include <algorithm>
#include <cmath>

namespace gubg { namespace ml { namespace cost { namespace bitsize { 

    template <typename T>
    class Entropy
    {
    public:
        T minimum_diff = 0.0;
        T overhead = 0.0;
        T pow = 0.0;

        T operator()(T prediction, T actual) const
        {
            const auto abs_diff = std::max(std::abs(prediction-actual), minimum_diff);
            if (pow == 0.0)
                return std::log2(abs_diff + overhead);
            else
                return std::pow(abs_diff, pow)*std::log2(abs_diff + overhead);
        }
    };

} } } } 

#endif
