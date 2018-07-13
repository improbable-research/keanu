package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import java.util.Collections;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantDoubleVertex extends DoubleVertex implements Differentiable {

    public ConstantDoubleVertex(DoubleTensor constant) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()),
            new Observation<>()
        );
        setValue(constant);
    }

    public ConstantDoubleVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    public ConstantDoubleVertex(double[] vector) {
        this(DoubleTensor.create(vector));
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex<?>, DualNumber> dualNumbers) {
        return new DualNumber(getValue(), Collections.emptyMap());
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return getValue();
    }

}
