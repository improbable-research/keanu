package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;

public interface SamplableWithShape<T extends Tensor<?>> {

    T sampleWithShape(long[] shape, KeanuRandom random);

    default T sampleWithShape(long[] shape) {
        return sampleWithShape(shape, KeanuRandom.getDefaultRandom());
    }
}
