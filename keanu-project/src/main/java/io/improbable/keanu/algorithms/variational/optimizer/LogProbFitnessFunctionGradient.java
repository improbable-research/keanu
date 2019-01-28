package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogProbFitnessFunctionGradient implements FitnessFunctionGradient {

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public LogProbFitnessFunctionGradient(ProbabilisticWithGradientGraph probabilisticWithGradientGraph) {
        this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
        this.onGradientCalculation = null;
    }

    public Map<? extends VariableReference, DoubleTensor> value(Map<VariableReference, DoubleTensor> values) {

        Map<? extends VariableReference, DoubleTensor> diffs = probabilisticWithGradientGraph.logProbGradients(values);

        if (onGradientCalculation != null) {
            onGradientCalculation.accept(values, diffs);
        }

        return diffs;
    }
}
