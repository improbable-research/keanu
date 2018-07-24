package io.improbable.keanu.distributions.continuous;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.distributions.discrete.UniformInt;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.BuilderParameterException;
import io.improbable.keanu.vertices.MissingParameterException;

public class DistributionOfType {
    private static Logger log = LoggerFactory.getLogger(DistributionOfType.class);

    private DistributionOfType() {
    }

    public static DiscreteDistribution binomial(DoubleTensor p, IntegerTensor n) {
        return new Binomial(p, n);
    }

    public static DiscreteDistribution binomial(List<NumberTensor<?,?>> inputs) {
        return construct(Binomial.class, inputs, 2);
    }

    public static ContinuousDistribution beta(List<DoubleTensor> inputs) {
        return construct(Beta.class, inputs, 4);
    }

    public static ContinuousDistribution beta(DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax) {
        return new Beta(alpha, beta, xMin, xMax);
    }

    public static ContinuousDistribution chiSquared(List<IntegerTensor> inputs) {
        return construct(ChiSquared.class, inputs, 1);
    }

    public static ContinuousDistribution chiSquared(IntegerTensor k) {
        return new ChiSquared(k);
    }

    public static ContinuousDistribution exponential(List<DoubleTensor> inputs) {
        return construct(Exponential.class, inputs, 2);
    }

    public static ContinuousDistribution exponential(DoubleTensor location, DoubleTensor lambda) {
        return new Exponential(location, lambda);
    }

    public static ContinuousDistribution gamma(List<DoubleTensor> inputs) {
        return construct(Gamma.class, inputs, 3);
    }

    public static ContinuousDistribution gamma(DoubleTensor location, DoubleTensor theta, DoubleTensor k) {
        return new Gamma(location, theta, k);
    }

    public static ContinuousDistribution gaussian(List<DoubleTensor> inputs) {
        return construct(Gaussian.class, inputs, 2);
    }

    public static ContinuousDistribution gaussian(DoubleTensor mu, DoubleTensor sigma) {
        return new Gaussian(mu, sigma);
    }

    public static ContinuousDistribution inverseGamma(List<DoubleTensor> inputs) {
        return construct(InverseGamma.class, inputs, 2);
    }

    public static ContinuousDistribution inverseGamma(DoubleTensor alpha, DoubleTensor beta) {
        return new InverseGamma(alpha, beta);
    }

    public static ContinuousDistribution laplace(List<DoubleTensor> inputs) {
        return construct(Laplace.class, inputs, 2);
    }

    public static ContinuousDistribution laplace(DoubleTensor mu, DoubleTensor beta) {
        return new Laplace(mu, beta);
    }

    public static ContinuousDistribution logistic(List<DoubleTensor> inputs) {
        return construct(Logistic.class, inputs, 2);
    }

    public static ContinuousDistribution logistic(DoubleTensor mu, DoubleTensor s) {
        return new Logistic(mu, s);
    }

    public static ContinuousDistribution logNormal(List<DoubleTensor> inputs) {
        return construct(LogNormal.class, inputs, 2);
    }

    public static ContinuousDistribution logNormal(DoubleTensor mu, DoubleTensor sigma) {
        return new LogNormal(mu, sigma);
    }

    public static ContinuousDistribution multivariateGaussian(List<DoubleTensor> inputs) {
        return construct(MultivariateGaussian.class, inputs, 2);
    }

    public static ContinuousDistribution multivariateGaussian(DoubleTensor mu, DoubleTensor covariance) {
        return new MultivariateGaussian(mu, covariance);
    }

    public static DiscreteDistribution poisson(DoubleTensor mu) {
        return new Poisson(mu);
    }

    public static DiscreteDistribution poisson(List<DoubleTensor> inputs) {
        return construct(Poisson.class, inputs, 1);
    }

    public static ContinuousDistribution smoothUniform(List<DoubleTensor> inputs) {
        return construct(SmoothUniform.class, inputs, 3);
    }

    public static ContinuousDistribution smoothUniform(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor edgeSharpness) {
        return new SmoothUniform(xMin, xMax, edgeSharpness);
    }

    public static ContinuousDistribution studentT(List<IntegerTensor> inputs) {
        return construct(StudentT.class, inputs, 1);
    }

    public static ContinuousDistribution studentT(IntegerTensor v) {
        return new StudentT(v);
    }

    public static ContinuousDistribution triangular(List<DoubleTensor> inputs) {
        return construct(Triangular.class, inputs, 3);
    }

    public static ContinuousDistribution triangular(DoubleTensor min, DoubleTensor max, DoubleTensor c) {
        return new Triangular(min, max, c);
    }

    public static ContinuousDistribution uniform(List<DoubleTensor> inputs) {
        return construct(Uniform.class, inputs, 2);
    }

    public static ContinuousDistribution uniform(DoubleTensor min, DoubleTensor max) {
        return new Uniform(min, max);
    }

    public static DiscreteDistribution uniformInt(List<IntegerTensor> inputs) {
        return construct(UniformInt.class, inputs, 2);
    }

    public static DiscreteDistribution uniformInt(IntegerTensor min, IntegerTensor max) {
        return new UniformInt(min, max);
    }

    private static <D extends Distribution<? extends Tensor>> D construct(Class<D> clazz, List<? extends NumberTensor<?,?>> inputs, int expectedNumInputs) {
        if (inputs.size() < expectedNumInputs) {
            throw new MissingParameterException(
                String.format(
                    "Not enough parameters - expected %d, got %d",
                    expectedNumInputs, inputs.size()));
        } else if (inputs.size() > expectedNumInputs) {
            throw new BuilderParameterException(
                String.format(
                    "Too many parameters - expected %d, got %d",
                    expectedNumInputs, inputs.size()));

        }
        try {
            return construct(clazz, inputs);
        } catch (IllegalAccessException | InvocationTargetException | InstantiationException e) {
            String message = String.format(
                "Failed to construct Distribution class %s with inputs %s",
                clazz.getSimpleName(), inputs);
            log.error(message, e);
            throw new IllegalArgumentException(message, e);
        }
    }

    private static <D extends Distribution<? extends Tensor>> D construct(Class<D> clazz, List<? extends NumberTensor<?,?>> inputs)
        throws IllegalAccessException, InvocationTargetException, InstantiationException {
        Constructor[] constructors = clazz.getDeclaredConstructors();
        Constructor constructor = constructors[0];
        return (D) constructor.newInstance(inputs.toArray());
    }
}
