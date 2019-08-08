package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class MultivariateGaussianVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

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
                                      @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu,
                                      @LoadVertexParam(COVARIANCE_NAME) Vertex<DoubleTensor, ?> covariance) {
        super(shape);
        checkValidMultivariateShape(mu.getShape(), covariance.getShape());

        this.mu = wrapIfNeeded(mu);
        this.covariance = wrapIfNeeded(covariance);
        setParents(mu, covariance);
    }

    /**
     * Matches a mu and covariance of some shape to a Multivariate Gaussian
     *
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the covariance matrix of the Multivariate Gaussian
     */
    @ExportVertexToPythonBindings
    public MultivariateGaussianVertex(Vertex<DoubleTensor, ?> mu, Vertex<DoubleTensor, ?> covariance) {
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
    public MultivariateGaussianVertex(Vertex<DoubleTensor, ?> mu, double covariance) {
        this(mu, ConstantVertex.of(DoubleTensor.eye(mu.getShape()[0]).times(covariance)));
    }

    public MultivariateGaussianVertex(double mu, double covariance) {
        this(new ConstantDoubleVertex(DoubleTensor.vector(mu)), oneByOneMatrix(covariance));
    }

    private static DoubleVertex oneByOneMatrix(double value) {
        return new ConstantDoubleVertex(DoubleTensor.create(value, new long[]{1, 1}));
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


    /**
     * Math credit to:
     * https://math.stackexchange.com/questions/1599966/derivative-of-multivariate-normal-distribution-wrt-mean-and-covariance
     *
     * @param x             at value
     * @param withRespectTo list of parents to differentiate with respect to
     * @return the derivative of the logProb wrt given vertices
     */
    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor x, Set<? extends Vertex> withRespectTo) {

        final DoubleTensor covInv = covariance.getValue().matrixInverse();
        final DoubleTensor xMinusMu = x.minus(mu.getValue());

        final long[] batchShape = ArrayUtils.subarray(x.getShape(), 0, x.getShape().length - 1);
        final long dims = x.getShape()[x.getShape().length - 1];

        final boolean wrtMu = withRespectTo.contains(mu);
        final boolean wrtX = withRespectTo.contains(this);

        Map<Vertex, DoubleTensor> diff = new HashMap<>();
        if (wrtMu || wrtX) {

            final DoubleTensor dLogProbWrtMuForBatch = covInv.matrixMultiply(
                xMinusMu.reshape(TensorShape.concat(batchShape, new long[]{dims, 1}))
            ).reshape(TensorShape.concat(batchShape, new long[]{dims}));

            if (wrtX) {
                final DoubleTensor dLogProbWrtX = dLogProbWrtMuForBatch
                    .sum(TensorShape.dimensionRange(
                        0,
                        dLogProbWrtMuForBatch.getShape().length - x.getShape().length
                    )).unaryMinus();
                diff.put(this, dLogProbWrtX);
            }

            if (wrtMu) {
                final DoubleTensor dLogProbWrtMu = dLogProbWrtMuForBatch
                    .sum(TensorShape.dimensionRange(
                        0,
                        dLogProbWrtMuForBatch.getShape().length - mu.getShape().length
                    ));
                diff.put(mu, dLogProbWrtMu);
            }
        }

        if (withRespectTo.contains(covariance)) {

            final DoubleTensor covInvTranspose = covInv.transpose();
            final DoubleTensor dLogProbWrtCovarianceForBatch = covInvTranspose.matrixMultiply(covInvTranspose)
                .times(xMinusMu.pow(2).sum(-1).reshape(TensorShape.concat(batchShape, new long[]{1, 1})))
                .minus(covInvTranspose)
                .times(0.5);

            final DoubleTensor dLogProbWrtCovariance = dLogProbWrtCovarianceForBatch
                .sum(TensorShape.dimensionRange(
                    0,
                    dLogProbWrtCovarianceForBatch.getShape().length - covariance.getShape().length
                ));

            diff.put(covariance, dLogProbWrtCovariance);
        }

        return diff;
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
