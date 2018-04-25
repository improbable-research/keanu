package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.SmoothUniformDistribution;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Random;

import static java.util.Collections.singletonMap;

public class SmoothUniformVertex extends ProbabilisticDouble {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final double edgeSharpness;
    private final Random random;

    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax, double edgeSharpness, Random random) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.edgeSharpness = edgeSharpness;
        this.random = random;
        setParents(xMin, xMax);
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax, Random random) {
        this(xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS, random);
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax, Random random) {
        this(new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS, random);
    }

    public SmoothUniformVertex(double xMin, double xMax, Random random) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS, random);
    }

    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(xMin, xMax, DEFAULT_EDGE_SHARPNESS, new Random());
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS, new Random());
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS, new Random());
    }

    public SmoothUniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS, new Random());
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    public double getEdgeSharpness() {
        return edgeSharpness;
    }

    @Override
    public double density(Double value) {
        final double min = xMin.getValue();
        final double max = xMax.getValue();
        final double shoulderWidth = this.edgeSharpness * (max - min);
        return SmoothUniformDistribution.pdf(min, max, shoulderWidth, value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        final double min = xMin.getValue();
        final double max = xMax.getValue();
        final double shoulderWidth = this.edgeSharpness * (max - min);
        final double dPdfdx = SmoothUniformDistribution.dPdfdx(min, max, shoulderWidth, this.getValue());

        return singletonMap(getId(), dPdfdx);
    }

    @Override
    public Double sample() {
        return SmoothUniformDistribution.sample(xMin.getValue(), xMax.getValue(), this.edgeSharpness, random);
    }

}
