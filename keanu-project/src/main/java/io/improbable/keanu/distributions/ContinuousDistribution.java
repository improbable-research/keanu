package io.improbable.keanu.distributions;

import java.util.List;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    List<DoubleTensor> dLogProb(DoubleTensor x);
}
