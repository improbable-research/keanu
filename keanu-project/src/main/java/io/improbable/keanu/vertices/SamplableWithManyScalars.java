package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;

public interface SamplableWithManyScalars<T extends Tensor<?>> extends Samplable<T>, SamplableWithShape<T>, HasShape {

    default T sampleManyScalars(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsAreScalar("Cannot sample many scalars from a non-scalar vertex", this.getShape());
        return sampleWithShape(shape, random);
    }

    default T sampleManyScalars(long[] shape) {
        return sampleManyScalars(shape, KeanuRandom.getDefaultRandom());
    }

    @Override
    default T sample(KeanuRandom random) {
        return sampleWithShape(getShape(), random);
    }
}
