package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Random;

import static java.util.Collections.singletonMap;

public class UniformVertex extends ProbabilisticDouble {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final Random random;

    public UniformVertex(DoubleVertex xMin, DoubleVertex xMax, Random random) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.random = random;
        setParents(xMin, xMax);
    }

    public UniformVertex(DoubleVertex xMin, double xMax, Random random) {
        this(xMin, new ConstantDoubleVertex(xMax), random);
    }

    public UniformVertex(double xMin, DoubleVertex xMax, Random random) {
        this(new ConstantDoubleVertex(xMin), xMax, random);
    }

    public UniformVertex(double xMin, double xMax, Random random) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), random);
    }

    public UniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(xMin, xMax, new Random());
    }

    public UniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, xMax, new Random());
    }

    public UniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax, new Random());
    }

    public UniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new Random());
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double density(Double value) {
        return Uniform.pdf(xMin.getValue(), xMax.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        double min = this.xMin.getValue();
        double max = this.xMax.getValue();

        if (this.getValue() <= min) {
            return singletonMap(getId(), Double.POSITIVE_INFINITY);
        } else if (this.getValue() >= max) {
            return singletonMap(getId(), Double.NEGATIVE_INFINITY);
        } else {
            return singletonMap(getId(), 0.0);
        }
    }

    @Override
    public Double sample() {
        return Uniform.sample(xMin.getValue(), xMax.getValue(), random);
    }


}
