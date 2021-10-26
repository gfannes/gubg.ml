#ifndef HEADER_gubg_ann_Layer_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Layer_hpp_ALREADY_INCLUDED

#include <gubg/ann/Neuron.hpp>
#include <gubg/ix/Range.hpp>
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

		void setup_io_ixs(ix::Range &input_ixr, ix::Range &output_ixr)
		{
			for (auto &neuron: neurons_)
				neuron.setup_io_ixs(input_ixr, output_ixr);
		}

		void setup_param_ixs(ix::Range &param_ixr)
		{
			for (auto &neuron: neurons_)
				neuron.setup_param_ixs(param_ixr);
		}

		template <typename Params, typename Activations, typename Sufficients>
		void forward(Params &&params, Activations &&activations, Sufficients &&sufficients) const
		{
			for (auto &neuron: neurons_)
				neuron.forward(params, activations, sufficients);
		}

		template <typename Params, typename Activations, typename Sufficients, typename Gradient, typename Errors>
		void backward(Params &&params, Activations &&activations, Sufficients &&sufficients, Gradient &&gradient, Errors &errors) const
		{
			for (auto &neuron: neurons_)
				neuron.backward(params, activations, sufficients, gradient, errors);
		}

	private:
		std::vector<Neuron> neurons_;
	};

} } 

#endif