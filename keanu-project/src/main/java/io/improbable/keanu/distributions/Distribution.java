package io.improbable.keanu.distributions;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface Distribution<T> {
    T sample(long[] shape, KeanuRandom random);
    DoubleTensor logProb(T x);
}
