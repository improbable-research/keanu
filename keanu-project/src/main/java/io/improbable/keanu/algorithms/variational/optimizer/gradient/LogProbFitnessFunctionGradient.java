package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public class LogProbFitnessFunctionGradient extends ProbabilityFitnessFunctionGradient {

    public LogProbFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                          BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation) {
        super(probabilisticModelWithGradient, onGradientCalculation);
    }

    public LogProbFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        super(probabilisticModelWithGradient, null);
    }

    @Override
    Map<? extends VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                      Map<VariableReference, DoubleTensor> values) {
        return probabilisticModelWithGradient.logProbGradients(values);
    }
}
