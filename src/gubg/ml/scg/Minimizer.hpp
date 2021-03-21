#ifndef HEADER_gubg_ml_scg_Minimizer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_scg_Minimizer_hpp_ALREADY_INCLUDED

#include <vector>
#include <cmath>
#include <numeric>

namespace gubg { namespace ml { namespace scg { 

    template <typename T>
    class Minimizer
    {
    public:
        template <typename Position, typename Gradient_func, typename Cost_func>
        bool update(Position &position, Gradient_func &&gradient_func, Cost_func &&cost_func)
        {
            MSS_BEGIN(bool);

            L(C(iteration_));
            ++iteration_;

            copy_(orig_position_, position);

            //1. Initialization
            const auto restart = (k_ == 0);
            if (restart)
            {
                L("1. Restarting algorithm");
                lambda_ = 0.000001;
                lambda_bar_ = 0.0;
                MSS(gradient_func(gradient_, position));
                conjugate_direction_ = gradient_;
                success_ = true;
            }
            else
            {
                L("1. Continuing");
                std::swap(gradient_, gradient_alt_);
                set_(conjugate_direction_, gradient_, conjugate_direction_, beta_);
            }
            ++k_;
            L("    " << "gradient: " << hr(gradient_));
            L("    " << "conj_dir: " << hr(conjugate_direction_));
            L("    " << C(lambda_)C(lambda_bar_));

            const auto ss_codi = inproduct_(conjugate_direction_, conjugate_direction_);
            const auto norm_codi = std::sqrt(ss_codi);
            L("    " << C(ss_codi)C(norm_codi));
            if (ss_codi < eps_ || norm_codi < eps_)
            {
                L("    Conjugate direction is too small, restarting algorithm");
                k_ = 0;
                return true;
            }

            const auto inprod_codi_grad = inproduct_(conjugate_direction_, gradient_);
            L("    " << C(inprod_codi_grad));

            //2. Calculate 2nd-order information
            if (success_)
            {
                const auto sigma_k = sigma_ / norm_codi;

                auto &position_alt = position;
                set_(position_alt, orig_position_, conjugate_direction_, -sigma_k);
                MSS(gradient_func(gradient_alt_, position_alt));

                //This gives different round-off errors compared to gubg::optimization::SCG but should be more efficient
                delta_ = inprod_codi_grad - inproduct_(conjugate_direction_, gradient_alt_);

                delta_ /= sigma_k;

                L("2. 2nd-order information: " << C(delta_)C(sigma_k));
            }

            //3. Scale delta_
            delta_ += (lambda_ - lambda_bar_)*ss_codi;
            L("3. Scaling: " << C(delta_));

            //4. Make Hessian positive-definite
            if (delta_ <= 0)
            {
                L("4. Making Hessian positive-definite");
                lambda_bar_ = 2*(lambda_ - delta_/ss_codi);
                delta_ = -delta_ + lambda_*ss_codi;
                lambda_ = lambda_bar_;
                L("    " << C(lambda_)C(lambda_bar_)C(delta_));
            }

            //5. Calculate step size
            const auto mu = inprod_codi_grad;
            const auto alpha = mu/delta_;
            L("5. Step size: " << C(mu)C(alpha));
            if (alpha*norm_codi <= eps_)
            {
                L("    Step is too small, restarting algorithm");
                k_ = 0;
                return true;
            }

            //6. Compute comparison parameter
            const auto cost = cost_func(orig_position_);
            set_(position, orig_position_, conjugate_direction_, -alpha);
            const auto cost_alt = cost_func(position);
            const auto diff = 2*delta_*(cost - cost_alt)/(mu*mu);
            L("6. Comparison parameter: " << C(diff));

            //7. Check for success
            success_ = (diff >= 0);
            if (success_)
            {
                L("7. Success");
                //position was already updated

                //gradient_alt_ will be swapped with gradient_ upon restart
                auto &gradient_new = gradient_alt_;
                MSS(gradient_func(gradient_new, position));

                lambda_bar_ = 0;

                if (k_ == position.size())
                {
                    L("    Restarting algorithm");
                    k_ = 0;
                    return true;
                }
                else
                {
                    const auto ss_grad_new = inproduct_(gradient_new, gradient_new);
                    if (ss_grad_new < eps_)
                    {
                        L("    Gradient is zero, restart algorithm");
                        k_ = 0;
                        return true;
                    }
                    beta_ = (ss_grad_new - inproduct_(gradient_alt_, gradient_))/mu;
                }

                if (diff >= 0.75)
                {
                    lambda_ *= 0.25;
                    L("    Reduced scale parameter: " << C(lambda_));
                }
            }
            else
            {
                L("7. No success");
                lambda_bar_ = lambda_;
                copy_(position, orig_position_);
            }

            //8. Increase the scale parameter
            if (diff < 0.25)
            {
                lambda_ += delta_*(1-diff)/ss_codi;
                L("8. Increased scale parameter: " << C(lambda_));
            }

            MSS_END();
        }

    private:
        using Vector = std::vector<T>;

        static void copy_(Vector &dst, const Vector &src)
        {
            dst = src;
        }
        static void set_(Vector &dst, const Vector &src, const Vector &direction, T factor)
        {
            const auto size = src.size();
            dst.resize(size);
            for (auto ix = 0u; ix < size; ++ix)
                dst[ix] = src[ix] + factor*direction[ix];
        }
        static T inproduct_(const Vector &a, const Vector &b)
        {
            return std::inner_product(a.begin(), a.end(), b.begin(), T{});
        }

        using Float = T;

        const Float eps_ = 0.0000000001;
        const Float sigma_ = 0.0001;

        unsigned int iteration_ = 0;

        unsigned int k_ = 0;
        Float lambda_{};
        Float lambda_bar_{};
        bool success_{};

        Float delta_{};
        Float beta_{};

        Vector gradient_;
        Vector gradient_alt_;
        Vector conjugate_direction_;
        Vector orig_position_;
    };

} } } 

#endif
