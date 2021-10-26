#ifndef HEADER_gubg_ann_Stack_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Stack_hpp_ALREADY_INCLUDED

#include <gubg/ann/Layer.hpp>
#include <gubg/ix/Range.hpp>

namespace gubg { namespace ann { 

	class Stack
	{
	public:
		struct Shape
		{
			std::size_t nr_inputs = 0u;
		};
		void setup(const Shape &shape)
		{
			shape_ = shape;
			current_nr_outputs_ = shape_.nr_inputs;
		}

		void add_layer(std::size_t nr_outputs, Transfer transfer)
		{
			auto &layer = layers_.emplace_back();
			layer.setup({.nr_inputs = current_nr_outputs_, .nr_outputs = nr_outputs, .transfer = transfer});
			current_nr_outputs_ = nr_outputs;
		}

		void setup_io_ixs(ix::Range &input_ixr, ix::Range &output_ixr)
		{
			for (auto ix = 0u; ix < layers_.size(); ++ix)
			{
				auto &layer = layers_[ix];

				if (ix == 0u)
				{
					input_ixr.resize(shape_.nr_inputs);
					layer.setup_io_ixs(input_ixr, output_ixr);
				}
				else
				{
					auto hidden_input_ixr = output_ixr;
					output_ixr.clear();
					layer.setup_io_ixs(hidden_input_ixr, output_ixr);
				}
			}
		}

		void setup_param_ixs(ix::Range &param_ixr)
		{
			for (auto &layer: layers_)
				layer.setup_param_ixs(param_ixr);
		}

		template <typename Params, typename Activations, typename Sufficients>
		void forward(Params &&params, Activations &&activations, Sufficients &&sufficients) const
		{
			for (const auto &layer: layers_)
				layer.forward(params, activations, sufficients);
		}

		template <typename Params, typename Activations, typename Sufficients, typename Gradient, typename Errors>
		void backward(Params &&params, Activations &&activations, Sufficients &&sufficients, Gradient &&gradient, Errors &&errors)
		{
			for (const auto &layer: layers_)
				layer.backward(params, activations, sufficients, gradient, errors);
		}

	private:
		Shape shape_;
		std::size_t current_nr_outputs_ = 0u;
		std::vector<Layer> layers_;
	};

} } 

#endif