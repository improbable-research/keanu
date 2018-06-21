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

        checkValidShape(mu, covariance);

        this.mu = mu;
        this.covariance = covariance;
        setParents(mu, covariance);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public MultivariateGaussian(DoubleVertex mu, DoubleVertex covariance) {
        this(mu.getShape(), mu, covariance);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return 0;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorMultivariateGaussian.sample(getShape(), mu.getValue(), covariance.getValue(), random);
    }

    private void checkValidShape(DoubleVertex mu, DoubleVertex covariance) {
        int[] covarianceShape = covariance.getShape();
        int[] muShape = mu.getShape();

        if (covarianceShape.length != 2 ||
            muShape.length != 2 ||
            covarianceShape[0] != covarianceShape[1] ||
            muShape[1] != 1 ||
            muShape[0] != covarianceShape[0]) {
            throw new IllegalArgumentException("Invalid sizing of parameters");
        }
    }
}
