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

		bool setup_ixrs(ix::Range &input_ixr, ix::Range_opt &output_ixr_opt, ix::Range &param_ixr)
		{
			MSS_BEGIN(bool);

			for (auto &neuron: neurons_)
				MSS(neuron.setup_ixrs(input_ixr, output_ixr_opt, param_ixr));

			MSS_END();
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
			for (const auto &neuron: neurons_)
				neuron.backward(params, activations, sufficients, gradient, errors);
		}

	private:
		std::vector<Neuron> neurons_;
	};

} } 

#endif