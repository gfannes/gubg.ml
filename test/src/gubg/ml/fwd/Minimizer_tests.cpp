#include <gubg/ml/fwd/Minimizer.hpp>
#include <gubg/Rosenbrock.hpp>
#include <gubg/hr.hpp>
#include <catch.hpp>
#include <array>
#include <iostream>
#include <fstream>
using namespace gubg::ml;

TEST_CASE("Gradient Descent tests", "[ut][ml][fwd][Minimizer]")
{
    fwd::Minimizer<double> minimizer;

    gubg::Rosenbrock<double> rb;

    std::array<double, 2> pos{-1.5,-1.5};

    std::ofstream fo{"fwd.gnuplot"};
    fo << "$data << EOD" << std::endl;

    const auto step_cnt = 2000;
    unsigned int count = 0;
    for (auto ix = 0u; ix < step_cnt; ++ix)
    {
        std::cout << C(ix) << gubg::hr(pos) << std::endl;
        fo << pos[0] << '\t' << pos[1]+ix*0.001 << std::endl;
        auto gradient = [&](auto &grad){
            ++count;
            return rb.gradient(grad, pos);
        };
        REQUIRE(minimizer.update(pos, gradient));
    }

    fo << "EOD" << std::endl;
    fo << "set xrange [-2:2]" << std::endl;
    fo << "set yrange [-2:2]" << std::endl;
    fo << "plot $data using 1:2 with lines" << std::endl;
    /* fo << "plot $data using 1:2 with points" << std::endl; */
    fo << "pause(-1)" << std::endl;

    std::cout << C(count) << std::endl;
}
