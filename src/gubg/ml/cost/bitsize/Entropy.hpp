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
            return value(prediction, actual);
        }

        T value(T prediction, T actual) const
        {
            const auto abs_diff_min = std::max(std::abs(prediction-actual), minimum_diff);
            if (pow == 0.0)
                return std::log2(abs_diff_min + overhead);
            else
                return std::pow(abs_diff_min, pow)*std::log2(abs_diff_min + overhead);
        }

        T derivative(T prediction, T actual) const
        {
            const auto diff = prediction-actual;
            const auto abs_diff = std::abs(diff);

            if (abs_diff <= minimum_diff)
                return 0.0;

            const auto abs_diff_min = std::max(abs_diff, minimum_diff);
            const auto sign_diff = (diff >= 0 ? 1.0 : -1.0);
            const auto der_log2 = sign_diff/std::log(2.0)/(abs_diff_min+overhead);

            if (pow == 0.0)
                return der_log2;

            return pow*std::pow(abs_diff_min, pow-1)*sign_diff*std::log2(abs_diff_min+overhead) + std::pow(abs_diff_min, pow)*der_log2;
        }
    };

} } } } 

#endif
