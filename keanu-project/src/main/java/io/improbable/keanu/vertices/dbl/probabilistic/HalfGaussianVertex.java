package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class HalfGaussianVertex extends GaussianVertex {

    private static final double MU_ZERO = 0.0;
    private static final double LOG_TWO = Math.log(2);

    /**
     * One sigma that matches a proposed tensor shape of HalfGaussian (a Gaussian with mu = 0 and
     * non-negative x).
     *
     * <p>If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param sigma the sigma of the HalfGaussian with either the same tensorShape as specified for
     *     this vertex or a scalar
     */
    public HalfGaussianVertex(int[] tensorShape, DoubleVertex sigma) {
        super(tensorShape, MU_ZERO, sigma);
    }

    public HalfGaussianVertex(DoubleVertex sigma) {
        super(MU_ZERO, sigma);
    }

    public HalfGaussianVertex(double sigma) {
        super(MU_ZERO, new ConstantDoubleVertex(sigma));
    }

    public HalfGaussianVertex(int[] tensorShape, double sigma) {
        super(tensorShape, MU_ZERO, new ConstantDoubleVertex(sigma));
    }

    @Override
    public double logProb(DoubleTensor value) {
        if (value.greaterThanOrEqual(MU_ZERO).allTrue()) {
            return super.logProb(value) + LOG_TWO * value.getLength();
        }
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return super.sample(random).absInPlace();
    }
}
