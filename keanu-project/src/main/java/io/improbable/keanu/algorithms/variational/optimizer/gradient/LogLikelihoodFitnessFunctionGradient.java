package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public class LogLikelihoodFitnessFunctionGradient extends ProbabilityFitnessFunctionGradient {

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation) {
        super(probabilisticModelWithGradient, onGradientCalculation);
    }

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        super(probabilisticModelWithGradient, null);
    }

    @Override
    Map<? extends VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                      Map<VariableReference, DoubleTensor> values) {
        return probabilisticModelWithGradient.logLikelihoodGradients(values);
    }
}