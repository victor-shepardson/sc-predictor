// PluginPredictor.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

//TODO: API should be: return prediction for next time step
// can compute error for last prediction externally with a delay
// or could have a mode switch or two diff UGens

#include "SC_PlugIn.hpp"
#include "Predictor.hpp"

static InterfaceTable* ft;

namespace Predictor {

// from https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

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

// TODO: move feature size to constructor

//idea: RNN reservoir feature

//idea: "running hallucination" / "resonator output":
// maintain a parallel feature vector and compute a feedback output from it,
// as with halluc=1, but keep training on the real input
// what would this do for feedback suppression? probably not much?
// this signal would have effectively random phase and smeared spectrum?


//idea: rollouts to predict N samples ahead

//idea: hallucination mode, feeding the last prediction instead of the input.
// needs some amplitude stabilization

//idea: probabilistic version so error is a proper measure of surprise

//idea: dropout for longer horizon with fixed cost.
// keep array of indices of size ACTIVE_FEATURES << FEATURE_SIZE
// mutate this array with PRNG while iterating over it in `predict`?
// better idea is probably to use IIR features ?

//idea: polyphase filter and multidimensional target, lower rate prediction

void Predictor::next(int nSamples) {
    const float* input = in(0);
    const float* learning_rate = in(1);
    const float* l1_penalty = in(2);
    const float* l2_penalty = in(3);
    const float* hallucinate = in(4);

    float* prediction = out(0);
    float* residual = out(1);
    float* sparsity = out(2);
    float* max_index = out(3);

    // note: this doesn't actually predict ahead and is only good for measuring
    // prediction error.
    // predicting ahead is slightly more complicated because previous prediction/features need 
    // to be kept to run the update later once target is available.
    // or could do: compute param update; do feat update; do predict; apply param update
    for (int i=0; i < nSamples; ++i) {
        float inp = input[i];
        float halluc = hallucinate[i];

        auto raw_pred = predict();
        auto pred = raw_pred + last; // predicting delta

        // if hallucinating, replace input with prediction
        inp += halluc*(pred-inp); 

        float target = inp - last; // predict delta

        float spars, idx;
        update_parameters(
            raw_pred, target, learning_rate[i], 
            l1_penalty[i], l2_penalty[i],
            spars, idx);

        prediction[i] = pred;
        residual[i] = inp - pred; 
        sparsity[i] = spars;
        max_index[i] = idx;

        last = inp;
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
    features[ring_ptr] = x / (1+fabs(x));
    // point to now-oldest
    ring_ptr = (ring_ptr+1) % FEATURE_SIZE;
}

void Predictor::update_parameters(
        float pred, float target, float lr, float l1, float l2,
        float &spars, float &idx) {
    auto err = (pred-target);
    float max_p = 0;
    float sum_p = 0;
    float max_idx = 0;

    const int min_delay = 32;

    float abs_p;

    for (size_t param_idx=0; param_idx<FEATURE_SIZE; param_idx++){
        auto feat_idx = (param_idx+ring_ptr)%FEATURE_SIZE;
        auto p = parameters[param_idx];
        parameters[param_idx] -= lr*(
            2*features[feat_idx]*err // minimize error
            + sgn(p)*l1 // weight decay
            + p*l2
            );
        if (FEATURE_SIZE - param_idx >= min_delay){
            abs_p = fabs(parameters[param_idx]);
            if (abs_p > max_p) {
                max_p = abs_p;
                max_idx = param_idx;
            }
        }
        // max_p = fmax(max_p, abs_p);
        sum_p += abs_p;
    }

    // gini coefficient-like sparsity measure
    spars = (max_p / sum_p * PARAM_SIZE - 1) / (PARAM_SIZE - 1);
    idx = FEATURE_SIZE - max_idx;
}

} // namespace Predictor

PluginLoad(PredictorUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<Predictor::Predictor>(ft, "Predictor", false);
}
