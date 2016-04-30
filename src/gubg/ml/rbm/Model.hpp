#ifndef HEADER_gubg_ml_rbm_Model_hpp_ALREADY_INCLUDED
#define HEADER_gubg_ml_rbm_Model_hpp_ALREADY_INCLUDED

#include "gubg/Matrix.hpp"
#include <vector>

namespace gubg { namespace ml { namespace rbm { 

    template <typename Visible, typename Weight>
        class Model
        {
            public:
                Model(size_t nr_visible, size_t nr_hidden): weights_v2h_(nr_hidden, nr_visible), bias_v_(nr_visible), bias_h_(nr_hidden)
            {
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
