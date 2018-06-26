package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorMultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Map;

public class MultivariateGaussian extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex covariance;

    public MultivariateGaussian(int[] shape, DoubleVertex mu, DoubleVertex covariance) {

        checkValidMultivariateShape(mu.getShape(), covariance.getShape());

        this.mu = mu;
        this.covariance = covariance;
        setParents(mu, covariance);
        setValue(DoubleTensor.placeHolder(shape));
    }

    //constructor where covariance is a scalar and we multiply by identity matrix

    public MultivariateGaussian(DoubleVertex mu, DoubleVertex covariance) {
        this(checkValidMultivariateShape(mu.getShape(), covariance.getShape()), mu, covariance);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor covarianceValues = covariance.getValue();

        return TensorMultivariateGaussian.logPdf(muValues, covarianceValues, value);
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        GaussianVertex gaussianVariates = new GaussianVertex(mu.getShape(), 0, 1);
        return TensorMultivariateGaussian.sample(mu.getValue(), covariance.getValue(), gaussianVariates, random);
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
