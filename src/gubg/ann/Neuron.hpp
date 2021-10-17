#ifndef HEADER_gubg_ann_Neuron_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Neuron_hpp_ALREADY_INCLUDED

#include <gubg/ann/Transfer.hpp>
#include <gubg/ix/Range.hpp>
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
		void setup(const Shape &shape) { shape_ = shape; }

		//Indicate this Neuron at what input index it can find its inputs
		//and where it should write its output
		template <typename Index>
		void consume_io(Index& input_ix, Index &output_ix)
		{
			input_ix_ = input_ix; input_ix += shape_.nr_inputs;
			output_ix_ = output_ix++;
		}

		//Indicate this Neuron at what parameter index it can find its parameters
		template <typename Index>
		void consume_params(Index& index)
		{
			bias_ix_ = index++;
			weight_ix = index; index += shape_.nr_inputs;
		}

		template <typename Inputs, typename Params, typename Outputs>
		auto forward(Inputs &&inputs, Params &&params, Outputs &&outputs) const
		{
			//Bias
			auto output = params[bias_ix_];

			//Input * Params
			const auto input = &inputs[input_ix_];
			const auto param = &params[weight_ix];
			for (auto ix = 0u; ix < shape_.nr_inputs; ++ix)
				output += input[ix]*param[ix];

			//Transfer
			switch (shape_.transfer)
			{
				case Transfer::Linear:    output = transfer::Linear::output(output); break;
				case Transfer::Tanh:      output = transfer::Tanh::output(output); break;
				case Transfer::Sigmoid:   output = transfer::Sigmoid::output(output); break;
				case Transfer::LeakyReLU: output = transfer::LeakyReLU::output(output); break;
				case Transfer::SoftPlus:  output = transfer::SoftPlus::output(output); break;
				case Transfer::Quadratic: output = transfer::Quadratic::output(output); break;
				default: assert(false);   output = 0; break;
			}

			outputs[output_ix_] = output;
		}

	private:
		Shape shape_;
		std::size_t input_ix_ = 0u;
		std::size_t bias_ix_ = 0u;
		std::size_t weight_ix = 0u;
		std::size_t output_ix_ = 0u;
	};

} } 

#endif