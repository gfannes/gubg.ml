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
			input_ix_ = input_ixr.start();

			if (!output_ixr_opt)
				output_ixr_opt = ix::Range{input_ixr.stop(), 0u};
			output_ix_ = output_ixr_opt->stop();
			output_ixr_opt->push_back(1);

			bias_ix_ = param_ixr.stop();
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
				for (auto ix0 = 0u; ix0 < shape_.nr_inputs; ++ix0)
					sufficient += param[ix0]*input[ix0];
			}

			sufficients[output_ix_] = sufficient;
			activations[output_ix_] = transfer::output(sufficient, shape_.transfer);
		}

		template <typename Params, typename Activations, typename Sufficients, typename Gradient, typename Errors>
		void backward(Params &&params, Activations &&activations, Sufficients &&sufficients, Gradient &&gradient, Errors &errors) const
		{
			auto error_deriv = errors[output_ix_]*transfer::derivative(sufficients[output_ix_], shape_.transfer);

			//Bias
			gradient[bias_ix_] += error_deriv;

			//Input * Params
			const auto input = &activations[input_ix_];
			auto error = &errors[input_ix_];
			const auto param = &params[weight_ix_()];
			auto grad = &gradient[weight_ix_()];
			for (auto ix0 = 0u; ix0 < shape_.nr_inputs; ++ix0)
			{
				grad[ix0] += error_deriv*input[ix0];
				error[ix0] += error_deriv*param[ix0];
			}
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
