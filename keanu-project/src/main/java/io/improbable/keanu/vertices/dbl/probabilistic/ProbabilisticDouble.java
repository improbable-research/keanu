package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticDouble extends DoubleVertex implements Probabilistic<DoubleTensor> {

    public ProbabilisticDouble() {
        super(new ProbabilisticValueUpdater<>());
    }

    public double logProb(double value) {
        return this.logProb(DoubleTensor.scalar(value));
    }
    public double logProb(double[] values) {
        return this.logProb(DoubleTensor.create(values));
    }

    public Map<Long, DoubleTensor> dLogProb(double value) {
        return this.dLogProb(DoubleTensor.scalar(value));
    }

    public Map<Long, DoubleTensor> dLogProb(double[] values) {
        return this.dLogProb(DoubleTensor.create(values));
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex<?>, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }

}
