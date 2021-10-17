#ifndef HEADER_gubg_ann_MLP_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_MLP_hpp_ALREADY_INCLUDED

#include <gubg/ann/Layer.hpp>

namespace gubg { namespace ann { 

	class MLP
	{
	public:
		struct Shape
		{
			std::size_t nr_inputs = 0u;
		};
		void setup(const Shape &shape)
		{
			current_nr_outputs_ = shape.nr_inputs;
		}

		void add_layer(std::size_t nr_outputs, Transfer transfer)
		{
			auto &layer = layers_.emplace_back();
			layer.setup({.nr_inputs = current_nr_outputs_, .nr_outputs = nr_outputs, .transfer = transfer});
			current_nr_outputs_ = nr_outputs;
		}

		//Indicate this MLP at what input index it can find its inputs
		//and where it should write its outputs
		template <typename Index>
		void consume_io(Index& input_ix, Index &output_ix)
		{
			Index tmp_input_ix = output_ix;
			for (auto ix = 0u; ix < layers_.size(); ++ix)
			{
				auto &layer = layers_[ix];

				if (ix == 0)
				{
					layer.consume_io(input_ix, output_ix);
				}
				else
				{
					layer.consume_io(tmp_input_ix, output_ix);
					tmp_input_ix = output_ix;
				}
			}
		}

		//Indicate this MLP at what parameter index it can find its parameters
		template <typename Index>
		void consume_params(Index& index)
		{
			for (auto &layer: layers_)
				layer.consume_params(index);
		}

		template <typename Inputs, typename Params, typename Outputs>
		auto forward(Inputs &&inputs, Params &&params, Outputs &&outputs) const
		{
			for (auto ix = 0u; ix < layers_.size(); ++ix)
			{
				auto &layer = layers_[ix];

				if (ix == 0)
					layer.forward(inputs, params, outputs);
				else
					layer.forward(outputs, params, outputs);
			}
		}

	private:
		std::size_t current_nr_outputs_ = 0u;
		std::vector<Layer> layers_;
	};

} } 

#endif