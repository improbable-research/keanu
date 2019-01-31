package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogLikelihoodFitnessFunctionGradient implements FitnessFunctionGradient {

    private final ProbabilisticModelWithGradient probabilisticModelWithGradient;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        this.probabilisticModelWithGradient = probabilisticModelWithGradient;
        this.onGradientCalculation = null;
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> getGradientsAt(Map<VariableReference, DoubleTensor> values) {

        Map<? extends VariableReference, DoubleTensor> diffs = probabilisticModelWithGradient.logLikelihoodGradients(values);

        if (onGradientCalculation != null) {
            onGradientCalculation.accept(values, diffs);
        }

        return diffs;
    }
}
