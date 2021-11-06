#ifndef HEADER_gubg_ann_Cost_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_Cost_hpp_ALREADY_INCLUDED

#include <gubg/ann/types.hpp>
#include <gubg/ix/Range.hpp>
#include <gubg/mss.hpp>
#include <cassert>

namespace gubg { namespace ann { 

	class Cost
	{
	public:
		enum Type {Quadratic};

		Cost &quadratic(double sigma)
		{
			type_ = Quadratic;
			sigma_ = sigma;
			factor_ = 0.5/(sigma_*sigma_);
			return *this;
		}

		bool setup_ixrs(const ix::Range &prediction_ixr, const ix::Range &target_ixr)
		{
			MSS_BEGIN(bool);

			MSS(prediction_ixr.size() == target_ixr.size());

			prediction_ixr_ = prediction_ixr;
			target_ixr_ = target_ixr;

			MSS_END();
		}

		template <typename Value, typename Prediction, typename Target>
		bool add_cost(Value &agg_cost, Prediction &&prediction, Target &&target) const
		{
			MSS_BEGIN(bool);

			MSS(!!type_);
			switch (*type_)
			{
				case Quadratic:
				{
					const auto size = prediction_ixr_.size();
					MSS(target_ixr_.size() == size);
					Value my_cost{};
					for (auto ix = 0u; ix < size; ++ix)
					{
						const auto p = prediction[prediction_ixr_[ix]];
						const auto t = target[target_ixr_[ix]];
						const auto diff = (p-t);
						my_cost += diff*diff*factor_;
					}

					agg_cost += my_cost;
				}
				break;

				default: MSS(false); break;
			}

			MSS_END();
		}


	private:
		std::optional<Type> type_;
		ix::Range prediction_ixr_;
		ix::Range target_ixr_;
		Float sigma_ = 0.0;
		Float factor_ = 0.0;
	};

} } 

#endif