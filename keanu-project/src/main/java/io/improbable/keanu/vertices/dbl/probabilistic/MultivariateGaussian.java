package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorMultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class MultivariateGaussian extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex covariance;

    public MultivariateGaussian(int[] shape, DoubleVertex mu, DoubleVertex covariance) {

        //check that mu is a vector
        //check that covariance is a matrix of size vector x vector
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape(), covariance.getShape());

        this.mu = mu;
        this.covariance = covariance;
        setParents(mu, covariance);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public MultivariateGaussian(DoubleVertex mu, DoubleVertex covariance) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), covariance.getShape()), mu, covariance);
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor covarianceValues = covariance.getValue();

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
}
