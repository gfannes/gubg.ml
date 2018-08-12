#include "catch.hpp"
#include "gubg/mlp/Structure.hpp"
#include "gubg/s11n.hpp"
using namespace gubg;

TEST_CASE("mlp::Structure tests", "[ut][mlp][Structure]")
{
    mlp::Structure mlp1(2);

    mlp1.add_layer(neural::Transfer::Sigmoid, 5, 0.5, 3.0);
    mlp1.add_layer(neural::Transfer::Linear, 1, 0.5, 3.0);

    std::string str1;
    REQUIRE(s11n::write_object(str1, ":mlp.Structure", mlp1));
    std::cout << C(str1) << std::endl;

    mlp::Structure mlp2;
    REQUIRE(s11n::read_object(str1, ":mlp.Structure", mlp2));

    std::string str2;
    REQUIRE(s11n::write_object(str2, ":mlp.Structure", mlp2));
    std::cout << C(str2) << std::endl;
    REQUIRE(str1 == str2);
}
