#ifndef HEADER_gubg_ann_Model_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Model_hpp_ALREADY_INCLUDED

#include <gubg/ann/types.hpp>
#include <gubg/ann/Stack.hpp>
#include <gubg/ann/Cost.hpp>

namespace gubg { namespace ann { 

	class Model
	{
	public:
		Cost prediction_cost;

		template <typename Value, typename DataFtor>
		bool avg_prediction_cost(Value &value, DataFtor &&data_ftor) const
		{
			MSS_BEGIN(bool);

			Float sum{};
			unsigned int count = 0;


			MSS(count > 0);
			value = sum/count;

			MSS_END();
		}

	private:
	};

} } 

#endif