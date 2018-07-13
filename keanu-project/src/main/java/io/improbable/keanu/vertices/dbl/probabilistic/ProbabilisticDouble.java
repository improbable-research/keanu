package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticDouble extends DoubleVertex implements Differentiable, Probabilistic<DoubleTensor> {

    public ProbabilisticDouble() {
        super(new ProbabilisticValueUpdater<>());
    }

    public double logPdf(double value) {
        return this.logProb(DoubleTensor.scalar(value));
    }

    public double logPdf(double[] values) {
        return this.logProb(DoubleTensor.create(values));
    }

    public Map<Long, DoubleTensor> dLogPdf(double value) {
        return this.dLogProb(DoubleTensor.scalar(value));
    }

    public Map<Long, DoubleTensor> dLogPdf(double[] values) {
        return this.dLogProb(DoubleTensor.create(values));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }
}
