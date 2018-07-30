package io.improbable.keanu.distributions;

import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.number.dbl.DoubleTensor;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    Diffs dLogProb(DoubleTensor x);
}
