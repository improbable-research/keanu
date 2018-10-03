package io.improbable.keanu.distributions;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public interface Distribution<T> {
    T sample(int[] shape, KeanuRandom random);

    DoubleTensor logProb(T x);
}
