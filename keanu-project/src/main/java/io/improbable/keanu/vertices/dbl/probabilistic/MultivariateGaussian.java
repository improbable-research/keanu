package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorMultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Map;

public class MultivariateGaussian extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex covariance;
    private final GaussianVertex variates;

    public MultivariateGaussian(int[] shape, DoubleVertex mu, DoubleVertex covariance) {

        checkValidMultivariateShape(mu.getShape(), covariance.getShape());

        this.mu = mu;
        this.covariance = covariance;
        this.variates = new GaussianVertex(mu.getShape(), 0, 1);
        setParents(mu, covariance);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public MultivariateGaussian(DoubleVertex mu, DoubleVertex covariance) {
        this(checkValidMultivariateShape(mu.getShape(), covariance.getShape()), mu, covariance);
    }

    public MultivariateGaussian(DoubleVertex mu, double covariance) {
        this(mu, createIdentityMatrixFromScalar(mu, covariance));
    }

    public MultivariateGaussian(double mu, double covariance) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(covariance));
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
        return TensorMultivariateGaussian.sample(mu.getValue(), covariance.getValue(), variates, random);
    }

    private static DoubleVertex createIdentityMatrixFromScalar(DoubleVertex mu, double covariance) {
        int dimensions = mu.getShape()[0];
        RealMatrix identityMatrix = MatrixUtils.createRealIdentityMatrix(dimensions).scalarMultiply(covariance);
        Nd4jDoubleTensor identityTensor = Nd4jDoubleTensor.create(identityMatrix);
        return new ConstantDoubleVertex(identityTensor);
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
