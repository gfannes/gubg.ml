#include "catch.hpp"
#include "gubg/mlp/Parameters.hpp"
#include "gubg/mlp/Structure.hpp"
#include "gubg/s11n.hpp"
using namespace gubg;

TEST_CASE("mlp::Parameters tests", "[ut][mlp][Parameters]")
{
    mlp::Structure mlp(2);
    mlp.add_layer(neural::Transfer::Sigmoid, 5, 0.5, 3.0);
    mlp.add_layer(neural::Transfer::Linear, 1, 0.5, 3.0);

    mlp::Parameters params1;
    params1.setup(mlp);

    std::string str1;
    REQUIRE(s11n::write_object(str1, "params", params1));
    std::cout << str1 << std::endl;

    mlp::Parameters params2;
    REQUIRE(s11n::read_object(str1, "params", params2));

    std::string str2;
    REQUIRE(s11n::write_object(str2, "params", params2));
    std::cout << str2 << std::endl;

    REQUIRE(str1 == str2);
}
