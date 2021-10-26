#ifndef HEADER_gubg_ann_Neuron_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Neuron_hpp_ALREADY_INCLUDED

#include <gubg/ann/Transfer.hpp>
#include <gubg/ix/Range.hpp>
#include <gubg/mss.hpp>
#include <cassert>

namespace gubg { namespace ann { 

	class Neuron
	{
	public:
		struct Shape
		{
			std::size_t nr_inputs = 0u;
			Transfer transfer = Transfer::Linear;
		};
		const Shape &shape() const {return shape_;}

		void setup(const Shape &shape) { shape_ = shape; }

		bool setup_ixrs(ix::Range &input_ixr, ix::Range_opt &output_ixr_opt, ix::Range &param_ixr)
		{
			MSS_BEGIN(bool);

			input_ixr.resize(shape_.nr_inputs);
			input_ix_ = input_ixr.begin();

			if (!output_ixr_opt)
				output_ixr_opt = ix::Range{input_ixr.end(), 0u};
			output_ix_ = output_ixr_opt->end();
			output_ixr_opt->push_back(1);

			bias_ix_ = param_ixr.end();
			param_ixr.push_back(1u + shape_.nr_inputs);
	
			MSS_END();
		}

		template <typename Params, typename Activations, typename Sufficients>
		void forward(Params &&params, Activations &&activations, Sufficients &&sufficients) const
		{
			//Bias
			auto sufficient = params[bias_ix_];

			//Params * Inputs
			{
				const auto param = &params[weight_ix_()];
				const auto input = &activations[input_ix_];
				for (auto ix = 0u; ix < shape_.nr_inputs; ++ix)
					sufficient += param[ix]*input[ix];
			}

			sufficients[output_ix_] = sufficient;
			activations[output_ix_] = transfer::output(sufficient, shape_.transfer);
		}

		template <typename Params, typename Activations, typename Sufficients, typename Gradient, typename Errors>
		void backward(Params &&params, Activations &&activations, Sufficients &&sufficients, Gradient &&gradient, Errors &errors) const
		{
			#if 0
			const auto derived_output = errors[output_ix_];

			//Bias
			gradient[bias_ix_] += derived_output;

			//Input * Params
			const auto input = &inputs[input_ix_];
			const auto param = &params[weight_ix_()];
			auto grad = &gradient[weight_ix_()];
			auto derived_input = &errors[input_ix_];
			for (auto ix = 0u; ix < shape_.nr_inputs; ++ix)
			{
				grad[ix] += derived_output*input[ix];
				derived_input[ix] += derived_output*param[ix];
			}
				output += input[ix]*param[ix];

			outputs[output_ix_] = transfer::output(output, shape_.transfer);
			#endif
		}

	private:
		Shape shape_;
		std::size_t input_ix_ = 0u;
		std::size_t output_ix_ = 0u;

		std::size_t bias_ix_ = 0u;
		std::size_t weight_ix_() const {return bias_ix_+1;}
	};

} } 

#endif
