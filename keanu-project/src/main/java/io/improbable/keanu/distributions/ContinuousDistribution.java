package io.improbable.keanu.distributions;

import io.improbable.keanu.distributions.dual.Duals;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    Duals dLogProb(DoubleTensor x);
}
