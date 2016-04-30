#ifndef HEADER_gubg_ml_rbm_Model_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_rbm_Model_hpp_ALREADY_INCLUDED

#include "gubg/Matrix.hpp"
#include "gubg/sigmoid.hpp"
#include "gubg/Range.hpp"
#include "gubg/mss.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

namespace gubg { namespace ml { namespace rbm { 

    template <typename Weight>
        class Model
        {
            public:
                Model(size_t nr_visible, size_t nr_hidden): weights_v2h_(nr_hidden, nr_visible), bias_v_(nr_visible), bias_h_(nr_hidden)
            {
            }

                template <typename Energy, typename Visible, typename Hidden>
                    bool energy(Energy &e, const Visible &vis, const Hidden &hid) const
                    {
                        MSS_BEGIN(bool);
                        MSS(weights_v2h_.multiply(e, hid, vis));
                        assert(vis.size() == bias_v_.size());
                        assert(hid.size() == bias_h_.size());
                        e = std::inner_product(RANGE(bias_v_), vis.begin(), e);
                        e = std::inner_product(RANGE(bias_h_), hid.begin(), e);
                        e = -e;
                        MSS_END();
                    }
                template <typename Prob, typename Vis, typename Hid>
                    bool prob_unnorm(Prob &p, const Vis &vis, const Hid &hid) const
                    {
                        MSS_BEGIN(bool);
                        MSS(energy(p, vis, hid));
                        p = std::exp(-p);
                        MSS_END();
                    }

                template <typename Hidden, typename Visible>
                    bool prob_hidden(Hidden &hid, const Visible &vis) const
                    {
                        MSS_BEGIN(bool);
                        MSS(weights_v2h_.multiply(hid, vis));
                        using H = typename Hidden::value_type;
                        std::transform(RANGE(bias_h_), hid.begin(), hid.begin(), [](Weight bias, H h){return gubg::sigmoid(h+bias);});
                        MSS_END();
                    }

                template <typename Visible, typename Hidden>
                    bool prob_visible(Visible &vis, const Hidden &hid) const
                    {
                        MSS_BEGIN(bool);
                        MSS(weights_v2h_.multiply_trans(vis, hid));
                        using V = typename Visible::value_type;
                        std::transform(RANGE(bias_v_), vis.begin(), vis.begin(), [](Weight bias, V v){return gubg::sigmoid(v+bias);});
                        MSS_END();
                    }

            private:
                using Matrix = gubg::Matrix<Weight>;
                using Weights = std::vector<Weight>;

                Matrix weights_v2h_;

                Weights bias_v_;
                Weights bias_h_;
        };

} } } 

#endif
