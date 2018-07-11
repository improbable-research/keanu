package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

import java.util.Map;

public abstract class ProbabilisticDouble extends DoubleVertex {

    public ProbabilisticDouble() {
        super(new ProbabilisticValueUpdater<>());
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }

}
