#include <gubg/ml/cost/Total.hpp>
#include <gubg/ml/cost/bitsize/Entropy.hpp>
#include <gubg/ml/cost/bitsize/OutOfRange.hpp>
#include <gubg/ml/cost/Quadratic.hpp>
#include <gubg/naft/Document.hpp>
#include <catch.hpp>
#include <vector>
using namespace gubg::ml;

TEST_CASE("ml::cost::Total tests", "[ut][ml][cost][Total]")
{
    using T = double;
    using Vec = std::vector<T>;

    cost::bitsize::Entropy<T> entropy;
    entropy.overhead = 2.0;
    entropy.pow = 0.5;
    auto entropy_cost = [&](auto &c, const auto &p, const auto &o){
        c = entropy(p, o);
        return true;
    };

    cost::bitsize::OutOfRange<T> oor{15, 64};
    auto oor_cost = [&](auto &c, const auto &p, const auto &o){
        c = oor(p, o);
        return true;
    };

    cost::Quadratic<T> quadratic;
    quadratic.factor = 1.0/200;
    auto quadratic_cost = [&](auto &c, const auto &p, const auto &o){
        c = quadratic(p, o);
        return true;
    };

    cost::Total<T, T> total_cost;
    total_cost.inputs = {0, 8,9,10,11,12, 40};
    total_cost.outputs = {0, 8,9,10,11,12, 40};

    gubg::naft::Document doc{std::cout};
    for (double x = -5; x < 45; x += 0.01)
    {
        T tc;
        auto predict = [&](auto &p, const auto &i){
            p = x;
            return true;
        };
        auto n = doc.node("Sample");
        REQUIRE(total_cost(tc, predict, entropy_cost));
        n.attr("x", x).attr("entropy", tc);
        REQUIRE(total_cost(tc, predict, oor_cost));
        n.attr("x", x).attr("oor", tc);
        REQUIRE(total_cost(tc, predict, quadratic_cost));
        n.attr("x", x).attr("quadratic", tc);
    }
}
