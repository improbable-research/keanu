package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Map;


public class MultivariateGaussianVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * Multivariate gaussian distribution. The shape is driven from mu, which must be a vector.
     * The shape of the covariance (matrix) must be a square that is the same height as mu.
     *
     * @param tensorShape      the desired shape of the vertex
     * @param mu         the mu of the Multivariate Gaussian
     * @param covariance the covariance matrix of the Multivariate Gaussian
     */
    // package private
    MultivariateGaussianVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex covariance) {
        super(false, tensorShape, DistributionOfType::multivariateGaussian, mu, covariance);
        checkValidMultivariateShape(mu.getShape(), covariance.getShape());
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

    private static int[] checkValidMultivariateShape(int[] muShape, int[] covarianceShape) {
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
