package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class HalfGaussianVertex extends GaussianVertex {

    private static final double LOG_TWO = Math.log(2);

    /**
     * One sigma that match a proposed tensor shape of Gaussian
     * <p>
     * If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param sigma       the sigma of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     */
    public HalfGaussianVertex(int[] tensorShape, DoubleVertex sigma) {
        super(tensorShape, 0.0, sigma);
    }

    public HalfGaussianVertex(DoubleVertex sigma) {
        super(0.0, sigma);
    }

    public HalfGaussianVertex(double sigma) {
        super(0.0, new ConstantDoubleVertex(sigma));
    }

    public HalfGaussianVertex(int[] tensorShape, double sigma) {
        super(tensorShape, 0.0, new ConstantDoubleVertex(sigma));
    }

    @Override
    public double logProb(DoubleTensor value) {
        if (value.greaterThanOrEqual(0.0).allTrue()) {
            return super.logProb(value) + LOG_TWO * value.getLength();
        }
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return super.sample(random).absInPlace();
    }
}
