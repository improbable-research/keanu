package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public interface SamplableWithShape<T extends Tensor<?>> {

    T sampleWithShape(long[] shape, KeanuRandom random);

    default T sampleWithShape(long[] shape) {
        return sampleWithShape(shape, KeanuRandom.getDefaultRandom());
    }
}
