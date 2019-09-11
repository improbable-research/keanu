package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.TensorShapeValidation;

public interface Samplable<T> extends HasShape {

    default T batchSample(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsAreScalar("Cannot sample many scalars from a non-scalar vertex", this.getShape());
        return sample(shape, random);
    }

    default T batchSample(long[] shape) {
        return batchSample(shape, KeanuRandom.getDefaultRandom());
    }

    default T sample(KeanuRandom random) {
        return sample(getShape(), random);
    }

    T sample(long[] shape, KeanuRandom random);

    default T sample(long[] shape) {
        return sample(shape, KeanuRandom.getDefaultRandom());
    }

    default T sample() {
        return sample(KeanuRandom.getDefaultRandom());
    }
}
