// PluginPredictor.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

//TODO: API should be: return prediction for next time step
// can compute error for last prediction externally with a delay
// or could have a mode switch or two diff UGens

#include "SC_PlugIn.hpp"
#include "Predictor.hpp"

static InterfaceTable* ft;

namespace Predictor {

Predictor::Predictor() {
    last = 0.0f;
    ring_ptr = 0;
    features = (float*)RTAlloc(this->mWorld, FEATURE_SIZE*sizeof(float));
    parameters = (float*)RTAlloc(this->mWorld, PARAM_SIZE*sizeof(float));
    memset(features, 0.0f, FEATURE_SIZE*sizeof(float));
    memset(parameters, 0.0f, PARAM_SIZE*sizeof(float));

    mCalcFunc = make_calc_function<Predictor, &Predictor::next>();
    next(1);
}

Predictor::~Predictor() {
    RTFree(this->mWorld, features);
    RTFree(this->mWorld, parameters);
}

// idea: dropout for longer horizon with fixed cost.
// keep array of indices of size ACTIVE_FEATURES << FEATURE_SIZE
// mutate this array with PRNG while iterating over it in predict

// better idea is probably to use IIR features ?

// idea: polyphase filter and multidimensional target, lower rate prediction

void Predictor::next(int nSamples) {
    const float* input = in(0);
    const float* learning_rate = in(1);
    const float* l1_penalty = in(2);

    float* outbuf = out(0);

    // note: this doesn't actually predict ahead and is only good for measuring
    // prediction error.
    // predicting ahead is slightly more complicated because previous prediction/features need 
    // to be kept to run the update later once target is available.
    // or could do: compute param update; do feat update; do predict; apply param update
    for (int i=0; i < nSamples; ++i) {
        // float target = input[i];
        float target = input[i] - last; // predict delta

        auto pred = predict();
        auto err = update_parameters(
            pred, target, learning_rate[i], l1_penalty[i]);
        outbuf[i] = pred + last;
        last = input[i];
        update_features(target);
    }
}

float Predictor::predict() {
    float acc = 0.0f;
    for (size_t param_idx=0; param_idx<FEATURE_SIZE; param_idx++){
        auto feat_idx = (param_idx+ring_ptr)%FEATURE_SIZE;
        acc += features[feat_idx]*parameters[param_idx];
    }
    return acc;
}

void Predictor::update_features(float x) {
    // replace oldest with newest
    features[ring_ptr] = x;
    // point to now-oldest
    ring_ptr = (ring_ptr+1) % FEATURE_SIZE;
}

float Predictor::update_parameters(float pred, float target, float lr, float reg) {
    auto err = (pred-target);

    for (size_t param_idx=0; param_idx<FEATURE_SIZE; param_idx++){
        auto feat_idx = (param_idx+ring_ptr)%FEATURE_SIZE;
        parameters[param_idx] -= lr*(
            2*features[feat_idx]*err // minimize error
            + float(parameters[param_idx] > 0)*reg // weight decay
            );
    }

    return err*err;
}

} // namespace Predictor

PluginLoad(PredictorUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<Predictor::Predictor>(ft, "Predictor", false);
}
