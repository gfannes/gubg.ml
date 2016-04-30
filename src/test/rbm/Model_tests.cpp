#include "catch.hpp"
#include "gubg/ml/rbm/Model.hpp"
#include <vector>
using namespace gubg::ml;
using namespace std;

TEST_CASE("Restricted Boltzmann Machine", "[ut][rbm]")
{
    size_t nr_vis = 10;
    size_t nr_hid = 3;

    rbm::Model<bool, bool> model(nr_vis, nr_hid);

    vector<bool> vis(nr_vis);
    vector<bool> hid(nr_hid);

    float energy;
    REQUIRE(model.energy(energy, vis, hid));
    REQUIRE(energy == 0.0);
}
