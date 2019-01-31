package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogLikelihoodFitnessFunction implements FitnessFunction {

    private final ProbabilisticModel probabilisticModel;
    private final BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation;

    public LogLikelihoodFitnessFunction(ProbabilisticModel probabilisticModel) {
        this.probabilisticModel = probabilisticModel;
        this.onFitnessCalculation = null;
    }

    @Override
    public double getFitnessAt(Map<VariableReference, DoubleTensor> values) {

        final double logProb = probabilisticModel.logLikelihood(values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(values, logProb);
        }

        return logProb;
    }
}
