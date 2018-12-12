package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;
import java.util.Set;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class HalfGaussianVertex extends GaussianVertex {

    private static final double MU_ZERO = 0.0;
    private static final double LOG_TWO = Math.log(2);

    /**
     * One sigma that matches a proposed tensor shape of HalfGaussian (a Gaussian with mu = 0 and non-negative x).
     * <p>
     * If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param sigma       the sigma of the HalfGaussian with either the same tensorShape as specified for this vertex or a scalar
     */
    public HalfGaussianVertex(long[] tensorShape, DoubleVertex sigma) {
        super(tensorShape, MU_ZERO, sigma);
    }

    @ExportVertexToPythonBindings
    public HalfGaussianVertex(@LoadVertexParam(SIGMA_NAME) DoubleVertex sigma) {
        super(MU_ZERO, sigma);
    }

    public HalfGaussianVertex(double sigma) {
        super(MU_ZERO, new ConstantDoubleVertex(sigma));
    }

    public HalfGaussianVertex(long[] tensorShape, double sigma) {
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
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Map<Vertex, DoubleTensor> logProb = super.dLogProb(value, withRespectTo);
        if (value.greaterThanOrEqual(MU_ZERO).allTrue()) {
            return logProb;
        } else {
            for (Map.Entry<Vertex, DoubleTensor> entry : logProb.entrySet()) {
                logProb.put(entry.getKey(), DoubleTensor.create(0.0, entry.getValue().getShape()));
            }
            return logProb;
        }
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return super.sampleWithShape(shape, random).absInPlace();
    }
}
