package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Triangular;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;
import java.util.Random;

public class TriangularVertex extends ProbabilisticDouble {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final DoubleVertex c;
    private final Random random;

    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c, Random random) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.c = c;
        this.random = random;
        setValue(sample());
        setParents(xMin, xMax, c);
    }

    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c) {
        this(xMin, xMax, c, new Random());
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double density(Double value) {
        return Triangular.pdf(xMin.getValue(), xMax.getValue(), c.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Double sample() {
        return Triangular.sample(xMin.getValue(), xMax.getValue(), c.getValue(), random);
    }

}
