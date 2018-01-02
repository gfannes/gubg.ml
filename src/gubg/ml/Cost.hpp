#ifndef HEADER_gubg_ml_Cost_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_Cost_hpp_ALREADY_INCLUDED

#include "gubg/mss.hpp"
#include <vector>

namespace gubg { namespace ml { 

    template <typename Input, typename Output, typename MeanModel, typename CostFunction>
    class Cost
    {
    public:
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        MeanModel mean_model;
        CostFunction cost_function;

        template <typename Float>
        bool operator()(Float &cost) const
        {
            MSS_BEGIN(bool);
            MSS(inputs.size() == outputs.size());
            const auto nr = inputs.size();
            MSS(nr > 0);
            //Assignment esures mean is properly sized
            Output mean = outputs[0];
            cost = 0.0;
            for (size_t i = 0; i < nr; ++i)
            {
                MSS(mean_model(mean, inputs[i]));
                Float one_cost;
                MSS(cost_function(one_cost, mean, outputs[i]));
                cost += one_cost;
            }
            cost /= double(nr);
            MSS_END();
        }

    private:
    };

} } 

#endif
