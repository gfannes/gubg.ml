#ifndef HEADER_gubg_ml_cost_Quadratic_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_cost_Quadratic_hpp_ALREADY_INCLUDED

namespace gubg { namespace ml { namespace cost { 

    template <typename T>
    class Quadratic
    {
    public:
        T factor = 1.0;

        T operator()(T prediction, T actual) const
        {
            const auto diff = prediction-actual;
            return diff*diff*factor;
        }
    private:
    };

} } } 

#endif
