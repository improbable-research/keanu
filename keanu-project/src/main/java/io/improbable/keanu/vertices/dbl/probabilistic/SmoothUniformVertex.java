package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.SmoothUniform;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Collections;
import java.util.Map;

import static java.util.Collections.singletonMap;

public class SmoothUniformVertex extends ProbabilisticDouble {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final double edgeSharpness;

    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax, double edgeSharpness) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.edgeSharpness = edgeSharpness;
        setParents(xMin, xMax);
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax, double edgeSharpness) {
        this(xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, double xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
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
    public double logPdf(Double value) {
        final double min = xMin.getValue();
        final double max = xMax.getValue();
        final double shoulderWidth = this.edgeSharpness * (max - min);
        final double density = SmoothUniform.pdf(min, max, shoulderWidth, value);
        return Math.log(density);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(Double value) {

        if (isObserved()) {
            return Collections.emptyMap();
        }

        final double min = xMin.getValue();
        final double max = xMax.getValue();
        final double shoulderWidth = this.edgeSharpness * (max - min);
        final double dPdfdx = SmoothUniform.dPdfdx(min, max, shoulderWidth, value);
        final double density = SmoothUniform.pdf(min, max, shoulderWidth, value);
        final double dlogPdfdx = dPdfdx / density;

        return singletonMap(getId(), DoubleTensor.scalar(dlogPdfdx));
    }

    @Override
    public Double sample(KeanuRandom random) {
        return SmoothUniform.sample(xMin.getValue(), xMax.getValue(), this.edgeSharpness, random);
    }

}
