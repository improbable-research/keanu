package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticWithGradientGraph;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogLikelihoodFitnessFunctionGradient implements FitnessFunctionGradient {

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticWithGradientGraph probabilisticWithGradientGraph) {
        this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
        this.onGradientCalculation = null;
    }

    public Map<? extends VariableReference, DoubleTensor> value(Map<VariableReference, DoubleTensor> values) {

        Map<? extends VariableReference, DoubleTensor> diffs = probabilisticWithGradientGraph.logLikelihoodGradients(values);

        if (onGradientCalculation != null) {
            onGradientCalculation.accept(values, diffs);
        }

        return diffs;
    }
}
