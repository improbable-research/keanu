package io.improbable.keanu.distributions;

import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    ParameterMap<DoubleTensor> dLogProb(DoubleTensor x);
}
