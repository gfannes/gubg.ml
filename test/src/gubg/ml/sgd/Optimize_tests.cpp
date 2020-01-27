#include <gubg/ml/sgd/Optimize.hpp>
#include <gubg/biquad/Filter.hpp>
#include <gubg/biquad/Tuner.hpp>
#include <gubg/math/constants.hpp>
#include <gubg/prob/Uniform.hpp>
#include <gubg/wav/Writer.hpp>
#include <gubg/hr.hpp>
#include <catch.hpp>
#include <cmath>

TEST_CASE("ml::sgd::Optimize tests", "[ut][ml][sgd][Optimize]")
{
    const double a = 1.0;
    const double b = 2.0;
    auto f = [&](double x, double y)
    {
        return (x-a)*(x-a) + (y-b)*(y-b);
    };
    auto g = [&](std::vector<double> &gradient, const std::vector<double> &xy)
    {
        MSS_BEGIN(bool);
        MSS(gradient.size() == 2);
        gradient[0] = 2.0*(xy[0]-a);
        gradient[1] = 2.0*(xy[1]-b);
        MSS_END();
    };

    gubg::ml::sgd::Optimize<double> optimize;
    optimize.rate = -0.5;

    std::vector<double> xy = {0.0, 0.0};

    for (auto ix = 0u; ix < 100; ++ix)
    {
        std::cout << xy[0] << " " << xy[1] << " " << f(xy[0], xy[1]) << std::endl;
        REQUIRE(optimize.update_nesterov(xy, g));
    }
}

TEST_CASE("ml::sgd::Optimize for biquad tests", "[ut][ml][sgd][Optimize][biquad]")
{
    struct Response
    {
        double freq = 0;
        double ampl = 0;
        Response(double freq, double ampl): freq(freq), ampl(ampl) {}
    };
    std::vector<Response> responses = {
        {20.0, 0.9},
        {100.0, 0.9},
        {150.0, 0.1},
        {350.0, 0.1},
    };
    using Biquad = gubg::biquad::Filter<double>;
    const double samplerate = 48000.0;
    gubg::wav::Writer ww("bq.wav", 1, samplerate);
    auto f = [&](const auto &vec, const Response &resp)
    {
        Biquad::Coefficients coeffs;
        coeffs.b0 = vec[0];
        coeffs.b1 = vec[1];
        coeffs.b2 = vec[2];
        coeffs.a1 = vec[3];
        coeffs.a2 = vec[4];

        Biquad bq;
        bq.set(coeffs);

        const double dT = 1.0/samplerate;
        double time = 0.0;


        auto process = [&]()
        {
            double ampl = 0.0;
            for (auto ix = 0u; ix < samplerate; ++ix, time += dT)
            {
                double v = std::sin(gubg::math::tau*resp.freq*time);
                v = bq(v);
                ww.add_value(v/2.0);
                ampl = std::max(ampl, std::abs(v));
                if (ampl >= 10.0)
                    return 10.0;
            }
            return ampl;
        };
        process();
        const auto ampl = process();
        /* std::cout << resp.freq << " " << resp.ampl << " " << ampl << std::endl; */
        return std::abs(ampl-resp.ampl);
    };
    auto avg = [&](const auto &vec)
    {
        auto avg = 0.0;
        for (const auto &resp: responses)
        {
            const auto ampl = f(vec, resp);
            std::cout << resp.freq << " " << ampl << std::endl;
            avg += ampl;
        }
        return avg/responses.size();
    };
    auto g = [&](std::vector<double> &gradient, const std::vector<double> &vec)
    {
        MSS_BEGIN(bool);
        MSS(gradient.size() == 5);
        auto &resp = gubg::prob::select_uniform(responses);
        const auto ampl = f(vec, resp);
        const auto eps = 0.000001;
        for (auto ix = 0u; ix < vec.size(); ++ix)
        {
            auto vv = vec;
            double ampl_pos, ampl_neg;
            if (false)
            {
                vv[ix] = vec[ix] + eps;
                ampl_pos = f(vv, resp);
                vv[ix] = vec[ix] - eps;
                ampl_neg = f(vv, resp);
            }
            else
            {
                vv[ix] = vec[ix] + eps;
                ampl_pos = avg(vv);
                vv[ix] = vec[ix] - eps;
                ampl_neg = avg(vv);
            }
            gradient[ix] = (ampl_pos-ampl_neg)/2.0/eps;
        }
        MSS_END();
    };

    gubg::ml::sgd::Optimize<double> optimize;
    optimize.rate = -0.0000001;

    std::vector<double> vec = {0.01, 0.02, 0.01, -0.01, 0.0};

    gubg::biquad::Tuner<double> tuner{48000};
    tuner.configure(125.0, 1.0, gubg::biquad::Type::LowPass);
    /* tuner.set_gain_db(-0.915); */
    const auto &coeffs = *tuner.compute();
    vec[0] = coeffs.b0/coeffs.a0;
    vec[1] = coeffs.b1/coeffs.a0;
    vec[2] = coeffs.b2/coeffs.a0;
    vec[3] = coeffs.a1/coeffs.a0;
    vec[4] = coeffs.a2/coeffs.a0;

    for (auto ix = 0u; ix < 10; ++ix)
    {
        std::cout << gubg::hr(vec) << avg(vec) << std::endl;
        break;
        REQUIRE(optimize.update_nesterov(vec, g));
    }
}
