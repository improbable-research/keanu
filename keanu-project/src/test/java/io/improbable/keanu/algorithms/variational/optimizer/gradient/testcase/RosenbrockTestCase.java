package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.List;

public class RosenbrockTestCase implements GradientOptimizationAlgorithmTestCase {

    private final double a;
    private final double b;

    public RosenbrockTestCase(double a, double b){
        this.a = a;
        this.b = b;
    }

    @Override
    public FitnessFunction getFitnessFunction() {

        return null;
    }

    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {
        return null;
    }

    @Override
    public List<? extends Variable> getVariables() {
        return null;
    }

    @Override
    public void assertResult(OptimizedResult result) {

    }
}
