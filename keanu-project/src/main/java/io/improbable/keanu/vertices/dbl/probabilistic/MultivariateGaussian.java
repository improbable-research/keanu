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

    public MultivariateGaussian(DoubleVertex mu, DoubleVertex covariance) {
        this(checkValidMultivariateShape(mu.getShape(), covariance.getShape()), mu, covariance);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor covarianceValues = covariance.getValue();

        DoubleTensor logPdfs = TensorMultivariateGaussian.logPdf(muValues, covarianceValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorMultivariateGaussian.sample(mu.getValue(), covariance.getValue(), random);
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
