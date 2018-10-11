package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public abstract class ProbabilisticIntegerVertex extends IntegerVertex implements ProbabilisticInteger {
    protected abstract IntegerTensor sampleWithShape(long[] shape, KeanuRandom random);

    @Override
    public final IntegerTensor sample(KeanuRandom random) {
        return sampleWithShape(this.getShape(), random);
    }

    public IntegerTensor sampleManyScalars(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsAreScalar("Cannot sample many scalars from a non-scalar vertex", this.getShape());
        return sampleWithShape(shape, random);
    }

    public final IntegerTensor sampleManyScalars(long[] shape) {
        return sampleManyScalars(shape, KeanuRandom.getDefaultRandom());
    }
}
