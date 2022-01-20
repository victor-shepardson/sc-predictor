// PluginPredictor.hpp
// Victor Shepardson (victor.shepardson@gmail.com)

#pragma once

#include "SC_PlugIn.hpp"

namespace Predictor {

class Predictor : public SCUnit {
public:
    Predictor();

    // Destructor
    // ~Predictor();

private:
    // Calc function
    void next(int nSamples);

    // Member variables
};

} // namespace Predictor
