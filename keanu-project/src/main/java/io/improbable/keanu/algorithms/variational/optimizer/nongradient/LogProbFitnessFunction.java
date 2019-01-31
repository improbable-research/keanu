package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogProbFitnessFunction implements FitnessFunction {

    private final ProbabilisticModel probabilisticModel;
    private final BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation;

    public LogProbFitnessFunction(ProbabilisticModel probabilisticModel) {
        this.probabilisticModel = probabilisticModel;
        this.onFitnessCalculation = null;
    }

    @Override
    public double getFitnessAt(Map<VariableReference, DoubleTensor> values) {

        double logProb = probabilisticModel.logProb(values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(values, logProb);
        }

        return logProb;
    }
}
