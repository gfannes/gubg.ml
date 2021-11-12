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
			std::size_t nr_outputs = 0u;
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

		bool setup_ixrs(ix::Range &input_ixr, ix::Range_opt &output_ixr_opt, ix::Range &param_ixr)
		{
			MSS_BEGIN(bool);

			MSS(current_nr_outputs_ == shape_.nr_outputs);
			MSS(!layers_.empty());

			for (auto ix = 0u; ix < layers_.size(); ++ix)
			{
				auto &layer = layers_[ix];

				if (ix == 0u)
				{
					input_ixr.resize(shape_.nr_inputs);
					MSS(layer.setup_ixrs(input_ixr, output_ixr_opt, param_ixr));
				}
				else
				{
					MSS(!!output_ixr_opt);
					auto hidden_input_ixr = *output_ixr_opt;
					output_ixr_opt.reset();
					MSS(layer.setup_ixrs(hidden_input_ixr, output_ixr_opt, param_ixr));
				}
			}

			MSS_END();
		}

		template <typename Params, typename Activations, typename Sufficients>
		void forward(Params &&params, Activations &&activations, Sufficients &&sufficients) const
		{
			for (const auto &layer: layers_)
				layer.forward(params, activations, sufficients);
		}

		template <typename Params, typename Activations, typename Sufficients, typename Gradient, typename Errors>
		void backward(Params &&params, Activations &&activations, Sufficients &&sufficients, Gradient &&gradient, Errors &&errors) const
		{
			for (auto ix = layers_.size(); ix > 0;)
			{
				--ix;
				const auto &layer = layers_[ix];

				layer.backward(params, activations, sufficients, gradient, errors);
			}
		}

	private:
		Shape shape_;
		std::size_t current_nr_outputs_ = 0u;
		std::vector<Layer> layers_;
	};

} } 

#endif