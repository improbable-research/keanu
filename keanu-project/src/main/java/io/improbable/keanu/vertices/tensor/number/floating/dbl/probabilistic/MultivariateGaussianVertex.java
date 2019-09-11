package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

/**
 * A full covariance matrix multivariate gaussian distribution. This vertex supports batching of all parameters.
 * <p>
 * If the covariance matrix is simply a diagonal matrix then it will be more efficient to use the GaussianVertex with
 * batched mu and sigma. E.g.
 * <p>
 * a MultivariateGaussianVertex with covariance
 * <p>
 * [1, 0]
 * [0, 2]
 * <p>
 * and mu
 * <p>
 * [-1, 2]
 * <p>
 * is the same as a GaussianVertex with sigma
 * <p>
 * [1, 2]
 * <p>
 * and mu
 * <p>
 * [-1, 2]
 */
public class MultivariateGaussianVertex extends VertexImpl<DoubleTensor, DoubleVertex>
    implements ProbabilisticDouble, Differentiable, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private final DoubleVertex covariance;
    private static final String MU_NAME = "mu";
    private static final String COVARIANCE_NAME = "covariance";

    /**
     * Multivariate gaussian distribution. Each parameter must have shapes that agree on the number of dimensions N.
     * E.g.
     * mu shape of (2) and therefore N = 2
     * covariance shape of (2, 2) and therefore N = 2
     * <p>
     * would have a result shape of (2)
     * <p>
     * This is the most common and most simple use case. More complex use cases can take advantage of batching.
     * <p>
     * Each parameter can be batched but if multiple parameters are batched then the batchShape must be broadcastable.
     * E.g.
     * <p>
     * mu shape of (2, 2) and therefore batchShape = (2) N = 2
     * covariance shape of (3, 1, 2, 2) and therefore batchShape = (3, 1), N = 2
     * <p>
     * would have a result shape of (3, 2, 2) and therefore batchShape = (3, 2) by broadcasting (2) with (3, 1)
     * <p>
     * If the result shape is (3, 2, 2) then the shape parameter here can further be used to batch. E.g. if the
     * shape parameter could be (4, 3, 2, 2) to further batch (4)
     *
     * @param shape      the desired shape of the vertex including any batching. Must be shape (batchShape, N),where N is the
     *                   number of dimensions and the batchShape can be any shape as long as it is broadcastable with
     *                   any other batched parameter.
     * @param mu         the mu of the Multivariate Gaussian. Must have shape (batchShape, N), where N is the
     *                   number of dimensions and the batchShape can be any shape as long as it is broadcastable with
     *                   any other batched parameter.
     * @param covariance the covariance matrix of the Multivariate Gaussian. Must have shape (batchShape, N, N) where
     *                   N is the number of dimensions. The batchShape can be any shape as long as it is broadcastable with
     *                   any other batched parameter.
     */
    public MultivariateGaussianVertex(@LoadShape long[] shape,
                                      @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu,
                                      @LoadVertexParam(COVARIANCE_NAME) Vertex<DoubleTensor, ?> covariance) {
        super(MultivariateGaussian.validateShapes(shape, mu.getShape(), covariance.getShape()));

        this.mu = wrapIfNeeded(mu);
        this.covariance = wrapIfNeeded(covariance);
        setParents(mu, covariance);
    }

    /**
     * Matches a mu and full covariance matrix of some shape to a Multivariate Gaussian distribution. Mu should
     * be shape (batchShape, N) where N is the number of dimensions and batchShape can be any shape that is broadcastable
     * with the covariance batchShape if it is also batched. The covariance matrix should be shape (batchShape, N, N) where
     * the batchShape must be broadcastable with the batchShape of mu. Only the lower triangle of the covariance matrix
     * is used due to it being assumed to be a symmetric matrix. The upper triangle will be ignored.
     *
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the covariance matrix of the Multivariate Gaussian
     */
    @ExportVertexToPythonBindings
    public MultivariateGaussianVertex(Vertex<DoubleTensor, ?> mu, Vertex<DoubleTensor, ?> covariance) {
        this(MultivariateGaussian.validateShapes(mu.getShape(), mu.getShape(), covariance.getShape()), mu, covariance);
    }

    /**
     * Matches a mu to a Multivariate Gaussian. The covariance value provided here
     * is used to create a covariance matrix by multiplying the scalar value against
     * an identity matrix of the appropriate size.
     *
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the scale of the identity matrix
     */
    public MultivariateGaussianVertex(Vertex<DoubleTensor, ?> mu, double covariance) {
        this(mu, ConstantVertex.of(DoubleTensor.eye(mu.getShape()[mu.getShape().length - 1]).times(covariance)));
    }

    public MultivariateGaussianVertex(DoubleTensor mu, DoubleTensor covariance) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(covariance));
    }

    /**
     * This treats the MultivariateGaussianVertex as effectively a GaussianVertex. Using the GaussianVertex will be
     * more efficient.
     *
     * @param mu         the mu for the distribution
     * @param covariance the value used to create a 1x1 covariance matrix
     */
    public MultivariateGaussianVertex(double mu, double covariance) {
        this(new ConstantDoubleVertex(DoubleTensor.vector(mu)), new ConstantDoubleVertex(DoubleTensor.create(covariance, new long[]{1, 1})));
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
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor x, Set<? extends Vertex> withRespectTo) {

        final boolean wrtX = withRespectTo.contains(this);
        final boolean wrtMu = withRespectTo.contains(mu);
        final boolean wrtCovariance = withRespectTo.contains(covariance);

        final MultivariateGaussian mvg = MultivariateGaussian.withParameters(mu.getValue(), covariance.getValue());
        final DoubleTensor[] dlogProb = mvg.dLogProb(x, wrtX, wrtMu, wrtCovariance);

        Map<Vertex, DoubleTensor> diff = new HashMap<>();

        if (dlogProb[0] != null) {
            diff.put(this, dlogProb[0]);
        }

        if (dlogProb[1] != null) {
            diff.put(mu, dlogProb[1]);
        }

        if (dlogProb[2] != null) {
            diff.put(covariance, dlogProb[2]);
        }

        return diff;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return MultivariateGaussian.withParameters(mu.getValue(), covariance.getValue()).sample(shape, random);
    }
}
