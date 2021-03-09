#ifndef HEADER_gubg_ml_fwd_Minimizer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_fwd_Minimizer_hpp_ALREADY_INCLUDED

#include <gubg/hr.hpp>
#include <gubg/mss.hpp>
#include <optional>
#include <vector>
#include <numeric>

namespace gubg { namespace ml { namespace fwd { 

    /* ForwardDescent minimizer */
    /* If the last 2 gradients point in the same direction, learning rate is increased, */
    /* else, it is decreased */
    template <typename T>
    class Minimizer
    {
    public:
        T learning_rate = 1.0;
        T forward_factor = 1.1;
        T backward_factor = 0.3;
        std::optional<unsigned int> max_decrease_count;

        //gradient_func() should compute the gradient in `position`
        template <typename Position, typename Gradient_func>
        bool update(Position &position, Gradient_func &&gradient_func)
        {
            MSS_BEGIN(bool, "");

            const auto size = position.size();

            gradient_.resize(size);

            orig_position_.resize(size);
            std::copy(position.begin(), position.end(), orig_position_.begin());

            if (prev_gradient_.empty())
            {
                prev_gradient_.resize(size);
                MSS(gradient_func(prev_gradient_));
            }
            L(C(gubg::hr(position)));
            L(C(gubg::hr(prev_gradient_)));

            auto set_proposal_position = [&](){
                for (auto ix = 0u; ix < size; ++ix)
                    position[ix] = orig_position_[ix] - learning_rate*prev_gradient_[ix];
                return gradient_func(gradient_);
            };

            T prev_inprod = 0;
            unsigned int same_inprod_count = 0;
            for (unsigned int iteration = 0; set_proposal_position() && same_inprod_count < 5 && (!max_decrease_count || iteration < *max_decrease_count); ++iteration)
            {
                const auto inprod = std::inner_product(gradient_.begin(), gradient_.end(), prev_gradient_.begin(), T{});
                L(C(gubg::hr(position)));
                L(C(gubg::hr(gradient_)));
                L(C(inprod));
                const bool is_forward = (inprod >= 0);

                if (is_forward)
                {
                    learning_rate *= forward_factor;
                    L("++ " << C(learning_rate));
                    break;
                }

                learning_rate *= backward_factor;
                L("-- " << C(learning_rate));

                if (std::abs(prev_inprod-inprod) < eps_)
                    ++same_inprod_count;
                else
                    same_inprod_count = 0;
                prev_inprod = inprod;
            }

            prev_gradient_ = gradient_;

            MSS_END();
        }
    private:
        const T eps_ = 0.0000000001;
        std::vector<T> gradient_;
        std::vector<T> prev_gradient_;
        std::vector<T> orig_position_;
    };

} } } 

#endif
