#ifndef HEADER_gubg_ann_optimization_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ann_optimization_hpp_ALREADY_INCLUDED

#include <gubg/ann/types.hpp>
#include <gubg/ix/Range.hpp>
#include <gubg/prob/Bernoulli.hpp>
#include <gubg/mss.hpp>
#include <optional>
#include <random>

//@todo: these optimization algo's are not ann-specific and should be moved out of the ann namespace
namespace gubg { namespace ann { namespace optimization { 

	class SteepestDescent
	{
	public:
		struct Params
		{
			Float learning_rate = 0.01;
		};
		Params params;

		void setup_ixr(const ix::Range &ixr) {ixr_ = ixr;}

		template <typename Location, typename Gradient>
		bool update(Location &&location, Gradient &&gradient) const
		{
			MSS_BEGIN(bool);

			MSS(!!ixr_);
			const auto &ixr = *ixr_;

			ixr.each([&](auto &loc, const auto &grad){loc += params.learning_rate*grad;},
				location, gradient);

			MSS_END();
		}

	private:
		std::optional<ix::Range> ixr_;
	};

	class Metropolis
	{
	public:
		struct Params
		{
			Float sigma;
		};

		template <typename Location>
		bool setup(Location &&location, const ix::Range &ixr, Float cost)
		{
			MSS_BEGIN(bool);

			ixr_ = ixr;
	
			MSS_END();
		}

		template <typename Location>
		bool update(Location &&proposal_location, Float proposal_cost)
		{
			MSS_BEGIN(bool);

			MSS(!!ixr_);
			const auto &ixr = *ixr_;

			//Should we accept the proposal from the previous iteration?
			bool accept = false;
			if (!accepted_opt_)
			{
				//First time, we always accept
				accept = true;
				auto &accepted = accepted_opt_.emplace();
				accepted.location.resize(ixr.end());
			}
			else if (proposal_cost <= accepted_opt_->cost)
			{
				//New cost is lower: always accept
				accept = true;
			}
			else
			{
				//Accept according to Metropolis-Hasting probabilities
				auto &accepted = *accepted_opt_;
				const auto cost_diff = proposal_cost-accepted.cost;
				bernoulli_.set_prob(std::exp(-cost_diff));
				accept = bernoulli_();
			}
			auto &accepted = *accepted_opt_;

			//Copy the proposal location and cost if accepted
			if (accept)
			{
				ixr.each([](auto &dst, auto src){dst = src;}, accepted.location, proposal_location);
				accepted.cost = proposal_cost;
			}

			//Create a new proposal location
			ixr.each([&](auto &dst, auto src){dst = src + normal_(rng_);}, proposal_location, accepted.location);

			MSS_END();
		}

	private:
		std::optional<ix::Range> ixr_;

		struct Info
		{
			std::vector<Float> location;
			Float cost = 0;
		};
		std::optional<Info> accepted_opt_;

		prob::Bernoulli bernoulli_;
        std::mt19937 rng_ = prob::rng();
        std::normal_distribution<Float> normal_{0.0f, 0.1f};
	};
	
} } } 

#endif