#include "catch.hpp"
#include "gubg/ml/rbm/Model.hpp"
using namespace gubg::ml;

TEST_CASE("Restricted Boltzmann Machine", "[ut][rbm]")
{
    rbm::Model<bool, bool> model(10, 3);
}
