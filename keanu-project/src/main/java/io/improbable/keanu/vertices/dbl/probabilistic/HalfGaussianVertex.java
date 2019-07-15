package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Set;

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
    public HalfGaussianVertex(@LoadShape long[] tensorShape,
                              @LoadVertexParam(SIGMA_NAME) Vertex<DoubleTensor, ?> sigma) {
        super(tensorShape, MU_ZERO, sigma);
    }

    @ExportVertexToPythonBindings
    public HalfGaussianVertex(Vertex<DoubleTensor, ?> sigma) {
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
        if (value.greaterThanOrEqual(MU_ZERO).allTrue().scalar()) {
            return super.logProb(value) + LOG_TWO * value.getLength();
        }
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex muPlaceholder = new DoublePlaceholderVertex(getMu().getShape());
        final DoublePlaceholderVertex sigmaPlaceholder = new DoublePlaceholderVertex(getSigma().getShape());

        final DoubleVertex gaussianLogProbOutput = Gaussian.logProbOutput(xPlaceholder, muPlaceholder, sigmaPlaceholder);

        final DoubleVertex result = gaussianLogProbOutput.plus(LOG_TWO);
        final DoubleVertex invalidMask = xPlaceholder.lessThanMask(MU_ZERO);
        final DoubleVertex halfGaussianLogProbOutput = result.setWithMask(invalidMask, Double.NEGATIVE_INFINITY).sum();

        // Set the value of muPlaceholder since we know it's 0.
        muPlaceholder.setValue(MU_ZERO);

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(getMu(), muPlaceholder)
            .input(getSigma(), sigmaPlaceholder)
            .logProbOutput(halfGaussianLogProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Map<Vertex, DoubleTensor> dLogProb = super.dLogProb(value, withRespectTo);
        if (value.greaterThanOrEqual(MU_ZERO).allTrue().scalar()) {
            return dLogProb;
        } else {
            for (Map.Entry<Vertex, DoubleTensor> entry : dLogProb.entrySet()) {
                DoubleTensor v = entry.getValue();
                dLogProb.put(entry.getKey(), v.setWithMaskInPlace(value.lessThanMask(DoubleTensor.scalar(MU_ZERO)), 0.0));
            }
            return dLogProb;
        }
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return super.sampleWithShape(shape, random).absInPlace();
    }
}
