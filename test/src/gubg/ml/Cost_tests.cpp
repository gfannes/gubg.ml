#include <gubg/ml/cost/Total.hpp>
#include <catch.hpp>
#include <array>
using namespace gubg;

namespace  { 
    using Input = std::array<double, 2>;
    using Output = double;
    struct MeanModel
    {
        bool operator()(Output &mean, const Input &input) const
        {
            mean = input[0]+input[1];
            return true;
        }
    };
    struct CostFunction
    {
        double sigma = 1.0;
        bool operator()(double &cost, Output mean, Output actual) const
        {
            const auto diff = (mean-actual);
            cost = diff*diff/sigma;
            return true;
        }
    };
} 

TEST_CASE("gubg::ml::cost::Total tests", "[ut][ml][cost][Total]")
{
    ml::cost::Total<Input, Output, MeanModel, CostFunction> total_cost;
    total_cost.inputs.push_back({1.0,2.0});
    total_cost.outputs.push_back(0.0);
    total_cost.inputs.push_back({2.0,2.0});
    total_cost.outputs.push_back(3.0);
    double c;
    REQUIRE(total_cost(c));
    REQUIRE(c == (9.0+1.0)/2.0);
}
