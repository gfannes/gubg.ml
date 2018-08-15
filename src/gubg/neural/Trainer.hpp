#ifndef HEADER_gubg_neural_Trainer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_Trainer_hpp_ALREADY_INCLUDED

#include "gubg/neural/Simulator.hpp"
#include "gubg/optimization/SCG.hpp"
#include "gubg/Range.hpp"
#include "gubg/mss.hpp"
#include "gubg/hr.hpp"
#include <vector>
#include <list>
#include <map>
#include <optional>
#include <numeric>

namespace gubg { namespace neural { 

    template <typename Float>
    class Trainer
    {
    public:
        using Simulator = neural::Simulator<Float>;

        Trainer(size_t input_size, size_t target_size): input_size_(input_size), target_size_(target_size)
        {
        }

        size_t data_size() const {return data_.size();}

        template <typename Input, typename Target>
        bool add(const Input &input, const Target &target)
        {
            MSS_BEGIN(bool);
            MSS(input.size() == input_size_);
            MSS(target.size() == target_size_);
            data_.emplace_back(input_size_, target_size_);
            auto &it = data_.back();
            it.first.assign(RANGE(input));
            it.second.assign(RANGE(target));
            MSS_END();
        }

        bool set(Simulator *simulator, size_t input, size_t output)
        {
            MSS_BEGIN(bool);
            MSS(!!simulator);
            simulator_ = simulator;
            input_ = input;
            output_ = output;
            states_.resize(simulator_->nr_states());
            preacts_.resize(simulator_->nr_states());
            derivative_.resize(simulator_->nr_states());
            gradient_.resize(simulator_->nr_weights());
            fixed_inputs_.clear();
            MSS_END();
        }
        void add_fixed_input(size_t ix, Float value)
        {
            fixed_inputs_[ix] = value;
        }

        void set_max_gradient_norm(std::optional<Float> max_gradient_norm) { max_gradient_norm_ = max_gradient_norm; } 

        template <typename LogProb>
        bool train_sd(LogProb &lp, Float *weights, Float output_stddev, Float weights_stddev, Float step)
        {
            MSS_BEGIN(bool);

            MSS(compute_gradient_(lp, weights, output_stddev, weights_stddev));

            const auto nr_weights = simulator_->nr_weights();
            for (size_t i = 0; i < nr_weights; ++i)
                weights[i] += step*gradient_[i];

            MSS_END();
        }

        struct AdamParams
        {
            Float alpha = 0.001;
            Float beta1 = 0.9;
            Float beta2 = 0.999;
            Float eps = 1.0e-8;
        };
        bool init_adam(const AdamParams &params = AdamParams{})
        {
            MSS_BEGIN(bool);
            MSS(!!simulator_);
            MSS(adam_state_.init(params, simulator_->nr_weights()));
            MSS_END();
        }
        template <typename LogProb>
        bool train_adam(LogProb &lp, Float *weights, Float output_stddev, Float weights_stddev)
        {
            MSS_BEGIN(bool);

            MSS(adam_state_.valid());

            ++adam_state_.t;

            MSS(compute_gradient_(lp, weights, output_stddev, weights_stddev));

            auto &m1 = adam_state_.m1;
            auto &m2 = adam_state_.m2;
            
            m1.update_moment1(gradient_);
            m2.update_moment2(gradient_);

            const auto size = simulator_->nr_weights();
            const auto alpha = adam_state_.alpha;
            for (size_t i = 0; i < size; ++i)
                weights[i] += alpha*m1.corrected(i)/(std::sqrt(m2.corrected(i))+adam_state_.eps);

            MSS_END();
        }

        void init_scg()
        {
            scg_.clear();
        }
        template <typename LogProb>
        bool train_scg(LogProb &lp, Float *weights, Float output_stddev, Float weights_stddev, unsigned int nr_iterations)
        {
            MSS_BEGIN(bool);

            control_.nr = nr_iterations;

            auto function = [&](const Weights &w){
                Float lp = 0.0;
                compute_output_(lp, w.data(), output_stddev, weights_stddev);
                return lp;
            };
            auto gradient = [&](Weights &g, const Weights &w){
                Float lp = 0.0;
                compute_gradient_(lp, w.data(), output_stddev, weights_stddev);
                g = gradient_;
            };

            const auto nr_weights = simulator_->nr_weights();
            Weights w(nr_weights);
            std::copy(weights, weights+nr_weights, w.data());

            lp = scg_(w, function, gradient);

            std::copy(RANGE(w), weights);

            MSS_END();
        }

    private:
        using IX = size_t;
        using Vector = std::vector<Float>;
        using Input = Vector;
        using Target = Vector;
        using IT = std::pair<Input, Target>;
        using Data = std::list<IT>;

        template <typename LogProb>
        bool compute_output_(LogProb &lp, const Float *weights, Float output_stddev, Float weights_stddev)
        {
            MSS_BEGIN(bool);

            MSS(!!simulator_);
            MSS(data_.size() > 0);
            MSS(output_stddev > 0.0);
            MSS(weights_stddev > 0.0);

            const Float output_factor = 1.0/output_stddev/output_stddev;

            for (const auto &p: fixed_inputs_)
                states_[p.first] = p.second;

            lp = 0.0;
            for (const auto &it: data_)
            {
                std::copy(RANGE(it.first), &states_[input_]);
                simulator_->forward(states_.data(), preacts_.data(), weights);
                Float ll = 0.0;
                for (size_t i = 0; i < target_size_; ++i)
                {
                    const Float diff = (states_[output_+i]-it.second[i]);
                    ll += diff*diff;
                }
                lp += ll;
            }
            lp *= -output_factor*0.5;
            lp /= data_.size();

            const auto nr_weights = simulator_->nr_weights();
            const Float weights_factor = 1.0/weights_stddev/weights_stddev;
            for (size_t i = 0; i < nr_weights; ++i)
                lp -= weights[i]*weights[i]*weights_factor*0.5;

            MSS_END();
        }

        template <typename LogProb>
        bool compute_gradient_(LogProb &lp, const Float *weights, Float output_stddev, Float weights_stddev)
        {
            MSS_BEGIN(bool);

            MSS(!!simulator_);
            MSS(data_.size() > 0);
            MSS(output_stddev > 0.0);
            MSS(weights_stddev > 0.0);

            const Float output_factor = 1.0/output_stddev/output_stddev;

            for (const auto &p: fixed_inputs_)
                states_[p.first] = p.second;

            lp = 0.0;
            std::fill(RANGE(gradient_), 0.0);
            for (const auto &it: data_)
            {
                std::copy(RANGE(it.first), &states_[input_]);
                simulator_->forward(states_.data(), preacts_.data(), weights);
                Float ll = 0.0;
                std::fill(RANGE(derivative_), 0.0);
                for (size_t i = 0; i < target_size_; ++i)
                {
                    const Float diff = (states_[output_+i]-it.second[i]);
                    derivative_[output_+i] = -diff*output_factor;
                    ll += diff*diff;
                }
                simulator_->backward(derivative_.data(), gradient_.data(), states_.data(), preacts_.data(), weights);
                lp += ll;
            }
            lp *= -output_factor*0.5;
            lp /= data_.size();

            const auto nr_weights = simulator_->nr_weights();
            const Float weights_factor = 1.0/weights_stddev/weights_stddev;
            for (size_t i = 0; i < nr_weights; ++i)
            {
                lp -= weights[i]*weights[i]*weights_factor*0.5;
                gradient_[i] = gradient_[i]/data_.size() - weights[i]*weights_factor;
            }

            //Rescale the gradient to max_gradient_norm_ when such a maximum
            //is given and the current gradient is too large
            if (max_gradient_norm_)
            {
                Float norm = 0.0;
                for (auto g: gradient_)
                    norm += g*g;
                norm = std::sqrt(norm);
                if (norm > *max_gradient_norm_)
                    for (auto &g: gradient_)
                        g /= norm;
            }

            MSS_END();
        }

        const size_t input_size_;
        const size_t target_size_;
        Data data_;

        Simulator *simulator_ = nullptr;
        IX input_;
        IX output_;
        long bias_ = -1;

        Vector states_;
        Vector preacts_;
        Vector derivative_;
        Vector gradient_;
        std::map<IX, Float> fixed_inputs_;
        std::optional<Float> max_gradient_norm_;

        struct AdamState
        {
            Float alpha;

            struct Moment
            {
                Float beta, one_min_beta, beta_pow_t, inv_one_min_beta_pow_t;
                Vector moment;
                bool init(Float b, size_t nr_weights)
                {
                    MSS_BEGIN(bool);
                    MSS(0.0 <= b && b < 1.0);
                    beta = b; one_min_beta = 1.0-b; beta_pow_t = 1.0, inv_one_min_beta_pow_t = 0.0;
                    moment.resize(nr_weights); std::fill(RANGE(moment), 0.0);
                    MSS_END();
                }
                template <typename Gradient>
                void update_moment1(const Gradient &gradient)
                {
                    for (size_t i = 0; i < gradient.size(); ++i)
                        moment[i] = beta*moment[i] + one_min_beta*gradient[i];
                    beta_pow_t *= beta;
                    inv_one_min_beta_pow_t = 1.0/(1.0-beta_pow_t);
                }
                template <typename Gradient>
                void update_moment2(const Gradient &gradient)
                {
                    for (size_t i = 0; i < gradient.size(); ++i)
                        moment[i] = beta*moment[i] + one_min_beta*gradient[i]*gradient[i];
                    beta_pow_t *= beta;
                    inv_one_min_beta_pow_t = 1.0/(1.0-beta_pow_t);
                }
                Float corrected(size_t ix) const
                {
                    return moment[ix]*inv_one_min_beta_pow_t;
                }
            };
            Moment m1;
            Moment m2;

            Float eps;


            long t = -1;

            bool valid() const { return t >= 0; }
            bool init(const AdamParams &params, size_t nr_weights)
            {
                MSS_BEGIN(bool);
                alpha = params.alpha;
                MSS(m1.init(params.beta1, nr_weights));
                MSS(m2.init(params.beta2, nr_weights));
                eps = params.eps;
                t = 0;
                MSS_END();
            }
        };
        AdamState adam_state_;

        using Weights = std::vector<Float>;
        struct Params
        {
            using Type = Weights;
            static Float sum_squares(const Weights &w)
            {
                return std::accumulate(RANGE(w), 0.0, [](Float sum, Float v){return sum + v*v;});
            }
            static Float inprod(const Weights &a, const Weights &b)
            {
                return std::inner_product(RANGE(a), b.begin(), 0.0);
            }
            static void update(Weights &dst, Float k, const Weights &src)
            {
                auto it = src.begin();
                for (auto &w: dst)
                    w += k*(*it++);
            }
            static unsigned int order(const Weights &w)
            {
                return w.size();
            }
        };
        struct Control
        {
            void scg_params(unsigned int iteration, Float lp, const Weights &w){}
            bool scg_terminate(unsigned int iteration, Float lp, const Weights &g)
            {
                return iteration+1 >= nr;
            }
            unsigned int nr = 1;
        };
        Control control_;
        optimization::SCG<Float, Params, Control> scg_{control_};
    };

} } 

#endif
