package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import lombok.Value;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

@Value
public class ApacheFitnessFunctionGradientAdaptor implements MultivariateVectorFunction {

    private final FitnessFunctionGradientFlatAdapter gradient;

    @Override
    public double[] value(double[] point) throws IllegalArgumentException {
        return gradient.gradient(point);
    }
}
