package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

public class MultivariateGaussianVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private final DoubleVertex covariance;
    private static final String MU_NAME = "mu";
    private static final String COVARIANCE_NAME = "covariance";

    /**
     * Multivariate gaussian distribution. The shape is driven from mu, which must be a vector.
     * The shape of the covariance (matrix) must be a square that is the same height as mu.
     *
     * @param shape      the desired shape of the vertex
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the covariance matrix of the Multivariate Gaussian
     */
    public MultivariateGaussianVertex(@LoadShape long[] shape,
                                      @LoadVertexParam(MU_NAME) DoubleVertex mu,
                                      @LoadVertexParam(COVARIANCE_NAME) DoubleVertex covariance) {
        super(shape);
        checkValidMultivariateShape(mu.getShape(), covariance.getShape());

        this.mu = mu;
        this.covariance = covariance;
        setParents(mu, covariance);
    }

    /**
     * Matches a mu and covariance of some shape to a Multivariate Gaussian
     *
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the covariance matrix of the Multivariate Gaussian
     */
    @ExportVertexToPythonBindings
    public MultivariateGaussianVertex(DoubleVertex mu, DoubleVertex covariance) {
        this(checkValidMultivariateShape(mu.getShape(), covariance.getShape()), mu, covariance);
    }

    /**
     * Matches a mu to a Multivariate Gaussian. The covariance value provided here
     * is used to create a covariance tensor by multiplying the scalar value against
     * an identity matrix of the appropriate size.
     *
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the scale of the identity matrix
     */
    public MultivariateGaussianVertex(DoubleVertex mu, double covariance) {
        this(mu, ConstantVertex.of(DoubleTensor.eye(mu.getShape()[0]).times(covariance)));
    }

    public MultivariateGaussianVertex(double mu, double covariance) {
        this(new ConstantDoubleVertex(DoubleTensor.vector(mu)), oneByOneMatrix(covariance));
    }

    private static DoubleVertex oneByOneMatrix(double value) {
        return new ConstantDoubleVertex(DoubleTensor.scalar(value).reshape(Tensor.ONE_BY_ONE_SHAPE));
    }

    @SaveVertexParam(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @SaveVertexParam(COVARIANCE_NAME)
    public DoubleVertex getCovariance() {
        return covariance;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor covarianceValues = covariance.getValue();

        return MultivariateGaussian.withParameters(muValues, covarianceValues).logProb(value).scalar();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex muPlaceholder = new DoublePlaceholderVertex(mu.getShape());
        final DoublePlaceholderVertex covPlaceholder = new DoublePlaceholderVertex(covariance.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(mu, muPlaceholder)
            .input(covariance, covPlaceholder)
            .logProbOutput(MultivariateGaussian.logProbGraph(xPlaceholder, muPlaceholder, covPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return MultivariateGaussian.withParameters(mu.getValue(), covariance.getValue()).sample(shape, random);
    }

    private static long[] checkValidMultivariateShape(long[] muShape, long[] covarianceShape) {

        if (covarianceShape.length != 2) {
            throw new IllegalArgumentException("Covariance must be matrix but was rank " + covarianceShape.length);
        }

        if (muShape.length != 1) {
            throw new IllegalArgumentException("Mu must be vector but was rank " + muShape.length);
        }

        if (covarianceShape[0] != covarianceShape[1]) {
            throw new IllegalArgumentException("Covariance matrix must be square. Given shape: " + Arrays.toString(covarianceShape));
        }

        if (muShape[0] != covarianceShape[0]) {
            throw new IllegalArgumentException("Dimension 0 of mu must equal dimension 0 of covariance. Given: mu " + muShape[0] + ", covariance " + covarianceShape[0]);
        }

        return muShape;
    }
}
