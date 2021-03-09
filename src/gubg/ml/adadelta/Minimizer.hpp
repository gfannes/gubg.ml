#ifndef HEADER_gubg_ml_adadelta_Minimizer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_adadelta_Minimizer_hpp_ALREADY_INCLUDED

#include <gubg/mss.hpp>
#include <vector>
#include <cmath>
#include <numeric>

namespace gubg { namespace ml { namespace adadelta { 

    class Minimizer
    {
    public:
        Minimizer(unsigned int dimension, double update_decay, double gradient_decay, double epsilon = 0.00000001):
            dimension_(dimension),
            update_decay_(update_decay),
            one_min_update_decay_(1.0-update_decay),
            gradient_decay_(gradient_decay),
            one_min_gradient_decay_(1.0-gradient_decay),
            epsilon_(epsilon),
            gradient_ss_(dimension_),
            update_ss_(dimension_),
            prev_gradient_(dimension_){}

        template <typename Position, typename Gradient>
        bool update(Position &position, const Gradient &gradient)
        {
            MSS_BEGIN(bool);

            MSS(position.size() == dimension_);
            MSS(gradient.size() == dimension_);

            const auto gradien_ss = std::inner_product(gradient.begin(), gradient.end(), gradient.begin(), 0.0);
            const auto prev_gradien_ss = std::inner_product(prev_gradient_.begin(), prev_gradient_.end(), prev_gradient_.begin(), 0.0);
            if (gradien_ss > epsilon_ && prev_gradien_ss > epsilon_)
            {
                /* const auto cos_angle = std::inner_product(prev_gradient_.begin(), prev_gradient_.end(), gradient.begin(), 0.0)/std::sqrt(gradien_ss)/std::sqrt(prev_gradien_ss); */
                auto cos_angle = std::inner_product(prev_gradient_.begin(), prev_gradient_.end(), gradient.begin(), 0.0)/gradien_ss;
                cos_angle = std::min(cos_angle, 1.0);

                /* learning_rate_ += meta_learning_rate_*std::inner_product(prev_gradient_.begin(), prev_gradient_.end(), gradient.begin(), 0.0)/gradien_ss; */
                learning_rate_ = meta_learning_rate_*std::exp(cos_angle);
                std::cout << cos_angle << " " << learning_rate_ << " " << gradien_ss << " " << prev_gradien_ss << std::endl;
            }
            std::copy(gradient.begin(), gradient.end(), prev_gradient_.begin());

            for (auto ix = 0u; ix < dimension_; ++ix)
            {
                auto &gradient_ss = gradient_ss_[ix];
                gradient_ss = gradient_decay_*gradient_ss + one_min_gradient_decay_*gradient[ix]*gradient[ix];

                auto &update_ss = update_ss_[ix];

                /* const double update = -learning_rate_*(std::sqrt(update_ss)+epsilon_)/(std::sqrt(gradient_ss)+epsilon_)*gradient[ix]; */
                const double update = -learning_rate_*gradient[ix];
                position[ix] += update;

                update_ss = update_decay_*update_ss + one_min_update_decay_*update*update;
            }

            MSS_END();
        }

    private:
        const unsigned int dimension_;
        const double update_decay_;
        const double one_min_update_decay_;
        const double gradient_decay_;
        const double one_min_gradient_decay_;
        const double epsilon_;
        double learning_rate_ = 0.00001;
        double meta_learning_rate_ = 0.0001;
        std::vector<double> gradient_ss_;
        std::vector<double> update_ss_;
        std::vector<double> prev_gradient_;
    };

} } } 

#endif
