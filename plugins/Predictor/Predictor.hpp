// PluginPredictor.hpp
// Victor Shepardson (victor.shepardson@gmail.com)

#pragma once

#include "SC_PlugIn.hpp"

namespace Predictor {

const size_t FEATURE_SIZE = 512;
const size_t PARAM_SIZE = FEATURE_SIZE;

class Predictor : public SCUnit {
public:
    Predictor();

    // Destructor
    ~Predictor();

private:
    // Calc function
    void next(int nSamples);

    void update_features(float x);
    float predict();
    // returns error
    void update_parameters(
        float pred, float target, float lr, float reg,
        float &spars, float &idx);    

    // Member variables
    float* parameters;
    size_t ring_ptr;    
    float* features;
    float last;
};

} // namespace Predictor
