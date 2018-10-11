package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class ProbabilisticDoubleVertex extends DoubleVertex implements ProbabilisticDouble {
    protected abstract DoubleTensor sampleWithShape(long[] shape, KeanuRandom random);

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return sampleWithShape(this.getShape(), random);
    }

    public DoubleTensor sampleManyScalars(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsAreScalar("Cannot sample many scalars from a non-scalar vertex", this.getShape());
        return sampleWithShape(shape, random);
    }

    public final DoubleTensor sampleManyScalars(long[] shape) {
        return sampleManyScalars(shape, KeanuRandom.getDefaultRandom());
    }
}
