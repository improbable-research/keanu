package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

public class FitnessFunction {

    private final ProbabilisticGraph probabilisticGraph;
    private final boolean useLikelihood;
    private final List<? extends Variable> latentVariables;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunction(ProbabilisticGraph probabilisticGraph,
                           boolean useLikelihood,
                           BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticGraph = probabilisticGraph;
        this.useLikelihood = useLikelihood;
        this.latentVariables = probabilisticGraph.getLatentVariables();
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunction(ProbabilisticGraph probabilisticGraph, boolean useLikelihood) {
        this(probabilisticGraph, useLikelihood, null);
    }

    public MultivariateFunction fitness() {
        return point -> {

            Map<VariableReference, DoubleTensor> values = convertFromPoint(point, latentVariables);

            double logOfTotalProbability = useLikelihood ?
                probabilisticGraph.logLikelihood(values) :
                probabilisticGraph.logProb(values);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

}