#include "catch.hpp"
#include "gubg/ml/scg/Optimize.hpp"
#include "gubg/Range.hpp"
#include <vector>
#include <algorithm>
#include <sstream>
using namespace gubg::ml;

namespace  { 
    struct Function
    {
        struct Input
        {
            using Type = std::vector<double>;

            const size_t nr;

            Input(size_t nr): nr(nr) {}

            unsigned int cardinality() const {return nr;}
            template <typename Inputs>
                bool check(const Inputs &inputs) const
                {
                    return inputs.size() == nr;
                }
            template <typename Inputs>
                bool zero(Inputs &inputs) const
                {
                    MSS_BEGIN(bool);
                    inputs.resize(nr);
                    std::fill(RANGE(inputs), 0.0);
                    MSS_END();
                }
            template <typename Dst, typename Src>
                bool assign(Dst &dst, const Src &src) const
                {
                    MSS_BEGIN(bool);
                    MSS(check(src));
                    MSS(zero(dst));
                    std::copy(RANGE(src), dst.begin());
                    MSS_END();
                }
             template <typename Dst, typename Src, typename Scale>
                bool translate(Dst &dst, const Src &src, const Scale scale) const
                {
                    MSS_BEGIN(bool);
                    MSS(check(dst));
                    MSS(check(src));
                    for (size_t ix = 0; ix < nr; ++ix)
                        dst[ix] += scale*src[ix];
                    MSS_END();
                }
             template <typename Dst, typename Scale>
                bool scale(Dst &dst, const Scale scale) const
                {
                    MSS_BEGIN(bool);
                    MSS(check(dst));
                    for (auto &d: dst)
                        d *= scale;
                    MSS_END();
                }
           template <typename T, typename A, typename B>
                bool inproduct(T &v, const A &a, const B &b) const
                {
                    MSS_BEGIN(bool);
                    MSS(check(a));
                    MSS(check(b));
                    v = std::inner_product(RANGE(a), b.begin(), 0.0);
                    MSS_END();
                }
           template <typename Inputs>
               std::string to_hr(const Inputs &inputs) const
               {
                   std::ostringstream oss;
                   for (const auto &v: inputs)
                       oss << v << ", ";
                   return oss.str();
               }
        };
        Input input;

        Function(size_t nr_inputs): input(nr_inputs) {}

        template <typename Inputs>
            bool output(double &v, const Inputs &inputs) const
            {
                MSS_BEGIN(bool);
                MSS(input.check(inputs));
                v = std::accumulate(RANGE(inputs), 0.0, [](double sum, double v){return sum + v + v*v;});
                MSS_END();
            }
        template <typename Gradient, typename Inputs>
            bool gradient(Gradient &grad, const Inputs &inputs) const
            {
                MSS_BEGIN(bool);
                MSS(input.assign(grad, inputs));
                for (auto &g: grad)
                    g = 1.0+2.0*g;
                MSS_END();
            }
    };
} 

TEST_CASE("scg::minimize tests", "[ut][scg]")
{
    S("test");

    Function function(3);
    Function::Input::Type inputs;
    double output;

    SECTION("inputs should match nr_inputs")
    {
        REQUIRE(!function.input.check(inputs));
        REQUIRE(!function.output(output, inputs));
    }

    REQUIRE(function.input.zero(inputs));
    SECTION("zero-ed inputs should be ok")
    {
        REQUIRE(function.input.check(inputs));
        output = 42.0;
        REQUIRE(function.output(output, inputs));
        REQUIRE(output == 0.0);
    }

    SECTION("scg::minimize")
    {
        REQUIRE(scg::minimize(inputs, function));
        L("minimum is reached at " << function.input.to_hr(inputs));
    }
}
