package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class HalfCauchyVertex extends CauchyVertex {

    private static final double LOG_TWO = Math.log(2);

    /**
     * One scale that matches a proposed tensor shape of Cauchy
     * <p>
     * If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param scale       the scale of the HalfCauchy with either the same tensorShape as specified for this vertex or a scalar
     */
    public HalfCauchyVertex(int[] tensorShape, DoubleVertex scale) {
        super(tensorShape, 0.0, scale);
    }

    public HalfCauchyVertex(int[] tensorShape, double scale) {
        super(tensorShape, 0.0, scale);
    }

    public HalfCauchyVertex(DoubleVertex scale) {
        super(0.0, scale);
    }

    public HalfCauchyVertex(double scale) {
        super(0.0, scale);
    }

    @Override
    public double logProb(DoubleTensor value) {
        if (value.greaterThanOrEqual(0.0).allTrue()) {
            return super.logProb(value) + LOG_TWO;
        }
        return Double.NEGATIVE_INFINITY;
    }

}
