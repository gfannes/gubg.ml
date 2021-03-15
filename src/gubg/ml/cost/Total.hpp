#ifndef HEADER_gubg_ml_cost_Total_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_cost_Total_hpp_ALREADY_INCLUDED

#include "gubg/mss.hpp"
#include <vector>

namespace gubg { namespace ml { namespace cost { 

    struct NoPredictor { };
    struct NoCost { };

    template <typename Input, typename Output, typename Predictor = NoPredictor, typename Cost = NoCost>
    class Total
    {
    public:
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        Predictor predictor;
        Cost cost;

        template <typename Float>
        bool operator()(Float &total_cost) const
        {
            MSS_BEGIN(bool);
            MSS(inputs.size() == outputs.size());
            const auto nr = inputs.size();
            MSS(nr > 0);
            //Set size via assignment; this handles both primitive output and std::vector<T>
            prediction_ = outputs[0];
            total_cost = 0.0;
            for (size_t i = 0; i < nr; ++i)
            {
                MSS(predictor(prediction_, inputs[i]));
                Float one_cost;
                MSS(cost(one_cost, prediction_, outputs[i]));
                total_cost += one_cost;
            }
            total_cost /= double(nr);
            MSS_END();
        }

        template <typename Float, typename Predictor_ftor, typename Cost_ftor>
        bool operator()(Float &total_cost, Predictor_ftor &&predictor_ftor, Cost_ftor &&cost_ftor) const
        {
            MSS_BEGIN(bool);

            MSS(inputs.size() == outputs.size());
            const auto nr = inputs.size();
            MSS(nr > 0);
            //Set size via assignment; this handles both primitive output and std::vector<T>
            prediction_ = outputs[0];
            total_cost = 0.0;
            for (size_t i = 0; i < nr; ++i)
            {
                MSS(predictor_ftor(prediction_, inputs[i]));
                Float one_cost;
                MSS(cost_ftor(one_cost, prediction_, outputs[i]));
                total_cost += one_cost;
            }
            total_cost /= double(nr);

            MSS_END();
        }

    private:
        mutable Output prediction_;
    };

} } } 

#endif
