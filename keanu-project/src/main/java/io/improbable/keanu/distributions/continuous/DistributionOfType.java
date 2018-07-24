package io.improbable.keanu.distributions.continuous;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

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

    public static ContinuousDistribution gaussian(List<DoubleTensor> inputs) {
        return construct(Gaussian.class, inputs, 2);
    }

    public static ContinuousDistribution gaussian(DoubleTensor mu, DoubleTensor sigma) {
        return new Gaussian(mu, sigma);
    }

    public static DiscreteDistribution poisson(DoubleTensor mu) {
        return new Poisson(mu);
    }

    public static DiscreteDistribution poisson(List<DoubleTensor> inputs) {
        return construct(Poisson.class, inputs, 1);
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
        Preconditions.checkArgument(inputs.size() == expectedNumInputs,
            "Too many input parameters - expected %s, got %s", expectedNumInputs, inputs.size());
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
