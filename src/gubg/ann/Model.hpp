#ifndef HEADER_gubg_ann_Model_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Model_hpp_ALREADY_INCLUDED

#include <gubg/ann/types.hpp>
#include <gubg/ann/Stack.hpp>
#include <gubg/ann/Cost.hpp>
#include <vector>
#include <algorithm>

namespace gubg { namespace ann { 

	class Model
	{
	public:
		Cost prediction_cost;

		template <typename Ftor>
		bool update_stack(Ftor &&ftor)
		{
			MSS_BEGIN(bool);

			ftor(stack_);

			input_ixr_.clear();
			output_ixr_opt_.reset();
			param_ixr_.clear();
			MSS(stack_.setup_ixrs(input_ixr_, output_ixr_opt_, param_ixr_));

			MSS(!!output_ixr_opt_);
			const auto &output_ixr = *output_ixr_opt_;

			const auto act_size = std::max(input_ixr_.end(), output_ixr.end());
			activations_.resize(act_size);
			sufficients_.resize(act_size);
			errors_.resize(act_size);

			params_.resize(param_ixr_.end());
			gradient_.resize(param_ixr_.end());

			MSS(prediction_cost.setup_ixrs(output_ixr, ix::Range{0u, output_ixr.size()}));

			MSS_END();
		}

		template <typename Value, typename DataYield>
		bool avg_cost(Value &value, DataYield &&data_yield) const
		{
			MSS_BEGIN(bool);

			MSS(!!output_ixr_opt_);
			const auto &output_ixr = *output_ixr_opt_;
			MSS(output_ixr.size() == 1);

			Float sum_cost{};
			unsigned int count = 0;

			bool ok = true;
			auto data_entry = [&](const Float *input, const Float *target){
				input_ixr_.each_with_offset([&](auto &activation, auto ix0){activation = input[ix0];}, activations_);
				stack_.forward(params_, activations_, sufficients_);
				ok = ok && prediction_cost.add_cost(sum_cost, activations_, target);
				++count;
			};
			data_yield(data_entry);
			MSS(ok);

			MSS(count > 0);
			value = sum_cost/count;

			MSS_END();
		}

		template <typename DataYield, typename GradientYield>
		bool avg_gradient(DataYield &&data_yield, GradientYield &&gradient_yield) const
		{
			MSS_BEGIN(bool);

			MSS(!!output_ixr_opt_);
			const auto &output_ixr = *output_ixr_opt_;
			MSS(output_ixr.size() == 1);

			param_ixr_.each([](auto &g){g = 0.0f;}, gradient_);

			bool ok = true;
			unsigned int count = 0;
			auto data_entry = [&](const Float *input, const Float *target){
				input_ixr_.each_with_offset([&](auto &activation, auto ix0){activation = input[ix0];}, activations_);
				stack_.forward(params_, activations_, sufficients_);

				ok = ok && prediction_cost.gradient(errors_, activations_, target);

				stack_.backward(params_, activations_, sufficients_, gradient_, errors_);

				++count;
			};
			data_yield(data_entry);
			MSS(ok);
			MSS(count > 0);

			param_ixr_.each([&](auto &g){g /= count;}, gradient_);

			gradient_yield(gradient_);

			MSS_END();
		}

	private:
		Stack stack_;
		ix::Range input_ixr_;
		std::optional<ix::Range> output_ixr_opt_;
		ix::Range param_ixr_;

		mutable std::vector<Float> activations_;
		mutable std::vector<Float> sufficients_;
		mutable std::vector<Float> errors_;

		std::vector<Float> params_;
		mutable std::vector<Float> gradient_;
	};

} } 

#endif