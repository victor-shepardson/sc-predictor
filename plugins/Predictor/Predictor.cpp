// PluginPredictor.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "SC_PlugIn.hpp"
#include "Predictor.hpp"

static InterfaceTable* ft;

namespace Predictor {

Predictor::Predictor() {
    mCalcFunc = make_calc_function<Predictor, &Predictor::next>();
    next(1);
}

void Predictor::next(int nSamples) {
    const float* input = in(0);
    const float* gain = in(1);
    float* outbuf = out(0);

    // simple gain function
    for (int i = 0; i < nSamples; ++i) {
        outbuf[i] = input[i] * gain[i];
    }
}

} // namespace Predictor

PluginLoad(PredictorUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<Predictor::Predictor>(ft, "Predictor", false);
}
