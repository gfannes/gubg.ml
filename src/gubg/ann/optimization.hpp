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

			previous.location.resize(ixr.end());
			ixr.each([](auto &dst, auto src){dst = src;}, previous.location, location);
			previous.cost = cost;

			proposal.location.resize(ixr.end());
			ixr.each([](auto &dst, auto src){dst = src;}, proposal.location, location);
			proposal.cost = cost;
	
			MSS_END();
		}

		template <typename Location>
		bool update(Location &&proposal_location, Float previous_proposal_cost)
		{
			MSS_BEGIN(bool);

			//Accept the proposal from the previous iteration according to the rules
			proposal.cost = previous_proposal_cost;
			if (previous_proposal_cost <= previous.cost)
			{
				MSS(accept_proposal_());
			}
			else
			{
				const auto cost_diff = previous_proposal_cost-previous.cost;
				bernoulli_.set_prob(std::exp(-cost_diff));
				if (bernoulli_())
					MSS(accept_proposal_());
			}

			//Create a new proposal location
			MSS(create_proposal_());

			MSS_END();
		}

	private:
		bool accept_proposal_()
		{
			MSS_BEGIN(bool);

			MSS(!!ixr_);
			const auto &ixr = *ixr_;
			ixr.each([](auto &dst, auto src){dst = src;}, previous.location, proposal.location);
			previous.cost = proposal.cost;

			MSS_END();
		}
		bool create_proposal_()
		{
			MSS_BEGIN(bool);

			MSS(!!ixr_);
			const auto &ixr = *ixr_;
			ixr.each([&](auto &dst, auto src){dst = src + normal_(rng_);}, proposal.location, previous.location);

			MSS_END();
		}

		std::optional<ix::Range> ixr_;

		struct Info
		{
			std::vector<Float> location;
			Float cost = 0;
		};
		Info previous;
		Info proposal;

		prob::Bernoulli bernoulli_;
        std::mt19937 rng_ = prob::rng();
        std::normal_distribution<Float> normal_{0.0f, 0.1f};
	};
	
} } } 

#endif