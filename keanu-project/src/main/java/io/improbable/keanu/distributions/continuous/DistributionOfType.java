package io.improbable.keanu.distributions.continuous;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class DistributionOfType {
    private static Logger log = LoggerFactory.getLogger(DistributionOfType.class);

    private DistributionOfType() {
    }

    public static ContinuousDistribution gaussian(List<DoubleTensor> inputs) {
        return construct(Gaussian.class, inputs, 2);
    }

    public static ContinuousDistribution gaussian(DoubleTensor mu, DoubleTensor sigma) {
        return new Gaussian(mu, sigma);
    }

    private static ContinuousDistribution construct(Class clazz, List<DoubleTensor> inputs, int expectedNumInputs) {
        Preconditions.checkArgument(inputs.size() == expectedNumInputs,
            "Too many input parameters - expected {}, got {}", expectedNumInputs, inputs.size());
        try {
            return construct(clazz, inputs);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException | InstantiationException e) {
            String message = String.format(
                "Failed to construct Distribution class %s with inputs %s",
                clazz.getSimpleName(), inputs);
            log.error(message, e);
            throw new IllegalArgumentException(message, e);
        }
    }

    private static ContinuousDistribution construct(Class clazz, List<DoubleTensor> inputs)
        throws NoSuchMethodException, IllegalAccessException, InvocationTargetException, InstantiationException {
        Constructor[] constructors = clazz.getDeclaredConstructors();
        Constructor constructor = constructors[0];
        return (ContinuousDistribution) constructor.newInstance(inputs.toArray());
    }
}
