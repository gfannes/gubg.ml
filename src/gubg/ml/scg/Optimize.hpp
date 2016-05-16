#ifndef HEADER_gubg_ml_scg_Optimize_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_scg_Optimize_hpp_ALREADY_INCLUDED

#include "gubg/mss.hpp"
#include <cmath>
#include <cassert>

namespace gubg { namespace ml { namespace scg { 

    template <typename Input, typename Function>
        bool minimize(Input &input, const Function &function)
        {
            MSS_BEGIN(bool, "scg");

            const double sigma = 1.0e-4;

            auto input_current = &input;
            Input input_b;
            auto input_next = &input_b;

            typename Function::Input::Type gradient_a, gradient_b;
            auto gradient_current = &gradient_a;
            MSS(function.gradient(*gradient_current, *input_current));
            MSS(function.input.scale(*gradient_current, -1.0));
            auto gradient_next = &gradient_b;

            typename Function::Input::Type direction_a, direction_b;
            auto direction_current = &direction_a;
            MSS(function.input.assign(*direction_current, *gradient_current));
            auto direction_next = &direction_b;

            typename Function::Input::Type s;

            double lamda_k = 1.0e-6;
            double lamda_raised = 0.0;
            double direction_l22;
            double delta_k;
            double mu_k;
            double alpha_k;
            double beta_k;
            double diff_k;

            auto too_small = [](double l2) { return l2 < 1.0e-8; };

            bool success = true;
            unsigned int k = 1;
            const unsigned int n = function.input.cardinality();

            double output_current, output_next;
            MSS(function.output(output_current, *input_current));

            while (true)
            {
                S("loop");
                L("Direction current: " << function.input.to_hr(*direction_current));
                MSS(function.input.inproduct(direction_l22, *direction_current, *direction_current));
                L(STREAM(k, direction_l22));
                if (too_small(direction_l22))
                {
                    if (&input != input_current)
                        function.input.assign(input, *input_current);
                    return true;
                }

                if (success)
                {
                    const auto direction_l2 = std::sqrt(direction_l22);
                    const double sigma_k = sigma/direction_l2;
                    L(STREAM(sigma_k));
                    MSS(function.input.assign(*input_next, *input_current));
                    MSS(function.input.translate(*input_next, *direction_current, sigma_k));
                    L("Input alt: " << function.input.to_hr(*input_next));
                    MSS(function.gradient(s, *input_next));
                    L("Gradient: " << function.input.to_hr(s));
                    MSS(function.input.translate(s, *gradient_current, -1.0));
                    L("Diff: " << function.input.to_hr(s));
                    MSS(function.input.scale(s, 1.0/sigma_k));
                    L("s: " << function.input.to_hr(s));
                    MSS(function.input.inproduct(delta_k, *direction_current, s));
                    L(STREAM(delta_k));
                }

                //3: Scale delta_k
                delta_k += (lamda_k-lamda_raised)*direction_l22;

                //4: Make Hessian positive definite
                if (delta_k <= 0)
                {
                    L("Making delta_k positive: " << STREAM(delta_k));
                    lamda_raised = 2.0*(lamda_k - delta_k/direction_l22);
                    delta_k = -delta_k + lamda_k*direction_l22;
                    lamda_k = lamda_raised;
                }
                L(STREAM(delta_k));
                assert(delta_k > 0);

                //5: Calculate step size
                MSS(function.input.inproduct(mu_k, *direction_current, *gradient_current));
                alpha_k = mu_k/delta_k;

                //6: Calculate the comparison parameter
                {
                    MSS(function.input.assign(*input_next, *input_current));
                    MSS(function.input.translate(*input_next, *direction_current, alpha_k));
                    MSS(function.output(output_next, *input_next));
                    diff_k = 2.0*delta_k*(output_current - output_next)/mu_k/mu_k;
                }

                if (diff_k >= 0.0)
                {
                    std::swap(input_current, input_next);
                    std::swap(output_current, output_next);
                    MSS(function.gradient(*gradient_next, *input_current));
                    MSS(function.input.scale(*gradient_next, -1.0));
                    lamda_raised = 0;
                    success = true;

                    MSS(function.input.assign(*direction_next, *gradient_next));
                    if (k == n)
                    {
                        //Restart: we take direction_next == gradient_next as the start for a new conjugate system
                        k = 1;
                    }
                    else
                    {
                        ++k;
                        MSS(function.input.translate(*gradient_current, *gradient_next, -1.0));
                        MSS(function.input.inproduct(beta_k, *gradient_current, *gradient_next));
                        MSS(function.input.translate(*direction_next, *direction_current, -beta_k));
                    }

                    std::swap(gradient_current, gradient_next);
                    std::swap(direction_current, direction_next);

                    if (diff_k >= 0.75)
                        lamda_k /= 4.0;
                }
                else
                {
                    lamda_raised = lamda_k;
                    success = false;
                }

                //8:
                if (diff_k < 0.25)
                {
                    //Increase scale parameter
                    lamda_k += delta_k*(1.0-diff_k)/direction_l22;
                }
            }

            MSS_END();
        }

} } } 

#endif
