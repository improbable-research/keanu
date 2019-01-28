package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.Map;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

@AllArgsConstructor
public class FitnessFunction implements MultivariateFunction {

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
    private final boolean useLikelihood;

    private final BiConsumer<double[], Double> onFitnessCalculation;

    @Override
    public double value(double[] point) {

        Map<VariableReference, DoubleTensor> values = getValues(point);

        double logOfTotalProbability = useLikelihood ?
            probabilisticWithGradientGraph.logLikelihood(values) :
            probabilisticWithGradientGraph.logProb(values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(point, logOfTotalProbability);
        }

        return logOfTotalProbability;
    }

    private Map<VariableReference, DoubleTensor> getValues(double[] point) {
        return convertFromPoint(point, probabilisticWithGradientGraph.getLatentVariables());
    }
}
