package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Collections;
import java.util.Map;

import static java.util.Collections.singletonMap;

public class UniformVertex extends ProbabilisticDouble {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;

    public UniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this.xMin = xMin;
        this.xMax = xMax;
        setParents(xMin, xMax);
    }

    public UniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double logPdf(Double value) {
        return Math.log(Uniform.pdf(xMin.getValue(), xMax.getValue(), value));
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(Double value) {

        if (isObserved()) {
            return Collections.emptyMap();
        }

        double min = this.xMin.getValue();
        double max = this.xMax.getValue();

        if (this.getValue() <= min) {
            return singletonMap(getId(), DoubleTensor.scalar(Double.POSITIVE_INFINITY));
        } else if (this.getValue() >= max) {
            return singletonMap(getId(), DoubleTensor.scalar(Double.NEGATIVE_INFINITY));
        } else {
            return singletonMap(getId(), DoubleTensor.scalar(0.0));
        }
    }

    @Override
    public Double sample(KeanuRandom random) {
        return Uniform.sample(xMin.getValue(), xMax.getValue(), random);
    }


}
