package io.improbable.keanu.distributions;

import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    Diffs dLogProb(DoubleTensor x);
}
