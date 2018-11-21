package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Map;
import java.util.Set;

public class MultivariateGaussianVertex extends DoubleVertex implements SaveableVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor> {

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
    public MultivariateGaussianVertex(long[] shape, DoubleVertex mu, DoubleVertex covariance) {
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
    public MultivariateGaussianVertex(@LoadParentVertex(MU_NAME) DoubleVertex mu,
                                      @LoadParentVertex(COVARIANCE_NAME) DoubleVertex covariance) {
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
        this(mu, ConstantVertex.of(DoubleTensor.eye(mu.getShape()[0])).times(covariance));
    }

    public MultivariateGaussianVertex(double mu, double covariance) {
        this(ConstantVertex.of(mu), ConstantVertex.of(covariance));
    }

    @SaveParentVertex(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @SaveParentVertex(COVARIANCE_NAME)
    public DoubleVertex getCovariance() {
        return covariance;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor covarianceValues = covariance.getValue();

        return io.improbable.keanu.distributions.continuous.MultivariateGaussian.withParameters(muValues, covarianceValues).logProb(value).scalar();
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
        if (covarianceShape.length != 2
            || muShape.length != 2
            || covarianceShape[0] != covarianceShape[1]
            || muShape[1] != 1
            || muShape[0] != covarianceShape[0]) {
            throw new IllegalArgumentException("Invalid sizing of parameters");
        } else {
            return muShape;
        }
    }
}
