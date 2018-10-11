package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class ProbabilisticBooleanVertex extends BoolVertex implements ProbabilisticBoolean {
    protected abstract BooleanTensor sampleWithShape(long[] shape, KeanuRandom random);

    @Override
    public final BooleanTensor sample(KeanuRandom random) {
        return sampleWithShape(this.getShape(), random);
    }

    public BooleanTensor sampleManyScalars(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsAreScalar("Cannot sample many scalars from a non-scalar vertex", this.getShape());
        return sampleWithShape(shape, random);
    }

    public final BooleanTensor sampleManyScalars(long[] shape) {
        return sampleManyScalars(shape, KeanuRandom.getDefaultRandom());
    }
}
