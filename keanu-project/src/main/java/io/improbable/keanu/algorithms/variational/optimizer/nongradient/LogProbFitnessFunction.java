package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public class LogProbFitnessFunction extends ProbabilityFitnessFunction {

    public LogProbFitnessFunction(ProbabilisticModel probabilisticModel,
                                  BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation) {
        super(probabilisticModel, onFitnessCalculation);
    }

    public LogProbFitnessFunction(ProbabilisticModel probabilisticModel) {
        super(probabilisticModel);
    }

    @Override
    double calculateFitness(ProbabilisticModel probabilisticModel, Map<VariableReference, DoubleTensor> values) {
        return probabilisticModel.logProb(values);
    }

}