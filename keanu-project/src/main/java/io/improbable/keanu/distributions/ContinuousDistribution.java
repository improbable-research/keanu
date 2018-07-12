package io.improbable.keanu.distributions;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;

public interface ContinuousDistribution extends Distribution<DoubleTensor> {
    public List<DoubleTensor> dLogProb(DoubleTensor x);
}
