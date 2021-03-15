#ifndef HEADER_gubg_ml_cost_bitsize_OutOfRange_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_cost_bitsize_OutOfRange_hpp_ALREADY_INCLUDED

#include <cmath>

namespace gubg { namespace ml { namespace cost { namespace bitsize { 

    template <typename T>
    class OutOfRange
    {
    public:
        OutOfRange(T threshold_small, T threshold_large): threshold_small_(threshold_small), threshold_large_(threshold_large)
        {
            bitwidth_small_ = std::log2(threshold_small_);
            bitwidth_large_ = bitwidth_small_ + std::log2(threshold_large_);
        }

        T operator()(T prediction, T actual) const
        {
            const auto abs_diff = std::abs(prediction-actual);
            return (abs_diff <= threshold_small_) ? bitwidth_small_ : bitwidth_large_;
        }
    private:
        T threshold_small_;
        T threshold_large_;
        T bitwidth_small_;
        T bitwidth_large_;
    };

} } } } 

#endif
