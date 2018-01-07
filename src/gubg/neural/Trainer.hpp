#ifndef HEADER_gubg_neural_Trainer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_neural_Trainer_hpp_ALREADY_INCLUDED

#include "gubg/neural/Network.hpp"
#include "gubg/Range.hpp"
#include "gubg/mss.hpp"
#include "gubg/hr.hpp"
#include <vector>
#include <list>
#include <map>

namespace gubg { namespace neural { 

    template <typename Float>
    class Trainer
    {
    public:
        using Network = neural::Network<Float>;

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

        bool set(Network *nn, size_t input, size_t output)
        {
            MSS_BEGIN(bool);
            MSS(!!nn);
            nn_ = nn;
            input_ = input;
            output_ = output;
            states_.resize(nn_->nr_states());
            preacts_.resize(nn_->nr_states());
            derivative_.resize(nn_->nr_states());
            gradient_.resize(nn_->nr_weights());
            fixed_inputs_.clear();
            MSS_END();
        }
        void add_fixed_input(size_t ix, Float value)
        {
            fixed_inputs_[ix] = value;
        }

        template <typename LogProb>
        bool train(LogProb &lp, Float *weights, Float output_stddev, Float weights_stddev, Float step)
        {
            MSS_BEGIN(bool);
            MSS(!!nn_);
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
                nn_->forward(states_.data(), preacts_.data(), weights);
                Float ll = 0.0;
                std::fill(RANGE(derivative_), 0.0);
                for (size_t i = 0; i < target_size_; ++i)
                {
                    const Float diff = (states_[output_+i]-it.second[i]);
                    derivative_[output_+i] = -diff*output_factor;
                    ll += diff*diff;
                }
                nn_->backward(derivative_.data(), gradient_.data(), states_.data(), preacts_.data(), weights);
                lp += ll;
            }
            lp *= -output_factor*0.5;
            lp /= data_.size();

            const auto nr_weights = nn_->nr_weights();
            const Float weights_factor = 1.0/weights_stddev/weights_stddev;
            for (size_t i = 0; i < nr_weights; ++i)
            {
                lp -= weights[i]*weights[i]*weights_factor*0.5;
                weights[i] += step*(gradient_[i]/data_.size() - weights[i]*weights_factor);
            }

            MSS_END();
        }

    private:
        using IX = size_t;
        using Vector = std::vector<Float>;
        using Input = Vector;
        using Target = Vector;
        using IT = std::pair<Input, Target>;
        using Data = std::list<IT>;

        const size_t input_size_;
        const size_t target_size_;
        Data data_;

        Network *nn_ = nullptr;
        IX input_;
        IX output_;
        long bias_ = -1;

        Vector states_;
        Vector preacts_;
        Vector derivative_;
        Vector gradient_;
        std::map<IX, Float> fixed_inputs_;
    };

} } 

#endif
