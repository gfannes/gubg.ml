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
			factor2_ = 1.0/(sigma_*sigma_);
			factor_ = 0.5*factor2_;
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
					prediction_ixr_.each_with_offset([&](auto p, auto ix0){
						const auto t = target[target_ixr_[ix0]];
						const auto diff = (p-t);
						my_cost += diff*diff*factor_;
					}, prediction);

					agg_cost += my_cost;
				}
				break;

				default: MSS(false); break;
			}

			MSS_END();
		}

		template <typename Gradient, typename Prediction, typename Target>
		bool gradient(Gradient &&grad, Prediction &&prediction, Target &&target) const
		{
			MSS_BEGIN(bool);

			MSS(!!type_);
			switch (*type_)
			{
				case Quadratic:
				{
					const auto size = prediction_ixr_.size();
					MSS(target_ixr_.size() == size);

					prediction_ixr_.each_with_offset([&](auto &g, auto p, auto ix0){
						const auto t = target[target_ixr_[ix0]];
						const auto diff = (p-t);
						g = diff*factor2_;
					}, grad, prediction);
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
		Float factor_ = 0.0;//1/(2*sigma_*sigma_)
		Float factor2_ = 0.0;//2*factor_
	};

} } 

#endif