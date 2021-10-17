#ifndef HEADER_gubg_ann_Layer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Layer_hpp_ALREADY_INCLUDED

#include <gubg/ann/Neuron.hpp>
#include <vector>

namespace gubg { namespace ann { 

	class Layer
	{
	public:
		struct Shape
		{
			std::size_t nr_inputs = 0u;
			std::size_t nr_outputs = 0u;
			Transfer transfer = Transfer::Linear;
		};
		void setup(const Shape &shape)
		{
			neurons_.resize(shape.nr_outputs);
			for (auto &neuron: neurons_)
				neuron.setup({.nr_inputs = shape.nr_inputs, .transfer = shape.transfer});
		}

		//Indicate this Layer at what input index it can find its inputs
		//and where it should write its outputs
		template <typename Index>
		void consume_io(Index& input_ix, Index &output_ix)
		{
			const auto orig_input_ix = input_ix;
			for (auto &neuron: neurons_)
			{
				input_ix = orig_input_ix;
				neuron.consume_io(input_ix, output_ix);
			}
		}

		//Indicate this Layer at what parameter index it can find its parameters
		template <typename Index>
		void consume_params(Index& index)
		{
			for (auto &neuron: neurons_)
				neuron.consume_params(index);
		}

		template <typename Inputs, typename Params, typename Outputs>
		auto forward(Inputs &&inputs, Params &&params, Outputs &&outputs) const
		{
			for (auto &neuron: neurons_)
				neuron.forward(inputs, params, outputs);
		}

	private:
		std::vector<Neuron> neurons_;
	};

} } 

#endif