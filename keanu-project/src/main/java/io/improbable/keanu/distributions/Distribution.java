package io.improbable.keanu.distributions;

import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.List;

public interface Distribution<T extends Tensor> {
    T sample(int[] shape, KeanuRandom random);
    DoubleTensor logProb(T x);
}
