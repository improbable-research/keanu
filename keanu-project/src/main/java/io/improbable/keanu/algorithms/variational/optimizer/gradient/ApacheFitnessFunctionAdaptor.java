package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import lombok.Value;
import org.apache.commons.math3.analysis.MultivariateFunction;

@Value
public class ApacheFitnessFunctionAdaptor implements MultivariateFunction {

    private final FitnessFunctionFlatAdapter fitnessFunction;

    @Override
    public double value(double[] point) {
        return fitnessFunction.fitness(point);
    }
}
