#ifndef HEADER_gubg_ml_sgd_Optimize_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_sgd_Optimize_hpp_ALREADY_INCLUDED

#include <gubg/mss.hpp>
#include <vector>

namespace gubg { namespace ml { namespace sgd { 

    //Original formulas:
    //Moving average of gradient:
    //  ma_gradient_ = beta*ma_gradient_ + (1-beta)*gradient(position+rate*beta*ma_gradient_)
    //  Note that Nesterov computes the grandient in position _after update with current gradient estimate, assuming this gradient will yield 0_.
    //Ascent according to ma_gradient_:
    //  position_ = position_ + rate*ma_gradient_
    //
    //Internally, we use this more optimal computation:
    //  v_ = rate*ma_gradient_
    //  v_ = rate*(beta*ma_gradient_ + (1-beta)*gradient)
    //     = beta*rate*ma_gradient_ + rate*(1-beta)*gradient(position+beta*rate*ma_gradient_)
    //     = beta*v_ + rate*(1-beta)*gradient(position+beta*v_)
    //  position_ = position_ + rate*ma_gradient_
    //            = position_ + v_

    template <typename T>
    class Optimize
    {
    public:
        using Position = std::vector<T>;

        T beta = 0.9;
        T rate = 0.1;

        Optimize() {}
        //rate should be negative when minimizing iso optimizing
        Optimize(T beta, T rate): beta(beta), rate(rate) {}

        //Stochastic gradient descent with Nesterov momentum
        template <typename Gradient>
        bool update_nesterov(Position &position, Gradient &&gradient)
        {
            MSS_BEGIN(bool);

            const auto size = position.size();

            v_.resize(size);
            tmp_position_.resize(size);
            for (auto ix = 0u; ix < size; ++ix)
            {
                v_[ix] *= beta;
                tmp_position_[ix] = position[ix] + v_[ix];
            }

            gradient_.resize(size);
            MSS(gradient(gradient_, tmp_position_));

            const auto stepsize = rate*(1.0-beta);
            for (auto ix = 0u; ix < size; ++ix)
            {
                v_[ix] += stepsize*gradient_[ix];
                position[ix] += v_[ix];
            }

            MSS_END();
        }

    private:
        std::vector<T> v_;
        std::vector<T> gradient_;
        Position tmp_position_;
    };

} } } 

#endif
