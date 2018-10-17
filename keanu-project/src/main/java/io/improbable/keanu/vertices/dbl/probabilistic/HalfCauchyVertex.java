package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class HalfCauchyVertex extends CauchyVertex {

    private static final double LOC_ZERO = 0.0;
    private static final double LOG_TWO = Math.log(2);

    /**
     * One scale that matches a proposed tensor shape of HalfCauchy (Cauchy with location = 0 and non-negative x)
     * <p>
     * If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param scale       the scale of the HalfCauchy with either the same tensorShape as specified for this vertex or a scalar
     */
    public HalfCauchyVertex(long[] tensorShape, DoubleVertex scale) {
        super(tensorShape, LOC_ZERO, scale);
    }

    public HalfCauchyVertex(long[] tensorShape, double scale) {
        super(tensorShape, LOC_ZERO, scale);
    }

    public HalfCauchyVertex(DoubleVertex scale) {
        super(LOC_ZERO, scale);
    }

    public HalfCauchyVertex(double scale) {
        super(LOC_ZERO, scale);
    }

    @Override
    public double logProb(DoubleTensor value) {
        if (value.greaterThanOrEqual(LOC_ZERO).allTrue()) {
            return super.logProb(value) + LOG_TWO * value.getLength();
        }
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return super.sample(random).absInPlace();
    }

}
