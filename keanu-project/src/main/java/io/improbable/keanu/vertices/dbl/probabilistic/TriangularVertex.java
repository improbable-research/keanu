package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Triangular;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

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
        setParents(xMin, xMax, c);
    }

    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, double c, Random random) {
        this(xMin, xMax, new ConstantDoubleVertex(c), random);
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, DoubleVertex c, Random random) {
        this(xMin, new ConstantDoubleVertex(xMax), c, random);
    }

    public TriangularVertex(double xMin, DoubleVertex xMax, DoubleVertex c, Random random) {
        this(new ConstantDoubleVertex(xMin), xMax, c, random);
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, double c, Random random) {
        this(xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c), random);
    }

    public TriangularVertex(double xMin, DoubleVertex xMax, double c, Random random) {
        this(new ConstantDoubleVertex(xMin), xMax, c, random);
    }

    public TriangularVertex(double xMin, double xMax, DoubleVertex c, Random random) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), c, random);
    }

    public TriangularVertex(double xMin, double xMax, double c, Random random) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c), random);
    }

    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, double c) {
        this(xMin, xMax, new ConstantDoubleVertex(c), new Random());
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, DoubleVertex c) {
        this(xMin, new ConstantDoubleVertex(xMax), c, new Random());
    }

    public TriangularVertex(double xMin, DoubleVertex xMax, DoubleVertex c) {
        this(new ConstantDoubleVertex(xMin), xMax, c, new Random());
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, double c) {
        this(xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c), new Random());
    }

    public TriangularVertex(double xMin, DoubleVertex xMax, double c) {
        this(new ConstantDoubleVertex(xMin), xMax, c, new Random());
    }

    public TriangularVertex(double xMin, double xMax, DoubleVertex c) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), c, new Random());
    }

    public TriangularVertex(double xMin, double xMax, double c) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c), new Random());
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
