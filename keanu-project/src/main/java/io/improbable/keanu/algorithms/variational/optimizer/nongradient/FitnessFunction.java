package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

public class FitnessFunction {

    private final ProbabilisticGraph probabilisticGraph;
    private final boolean useLikelihood;
    private final List<String> latentVariables;
    private final Map<String, long[]> latentShapes;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunction(ProbabilisticGraph probabilisticGraph,
                           boolean useLikelihood,
                           BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticGraph = probabilisticGraph;
        this.useLikelihood = useLikelihood;
        this.latentVariables = probabilisticGraph.getLatentVariables();
        this.latentShapes = probabilisticGraph.getLatentVariablesShapes();
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunction(ProbabilisticGraph probabilisticGraph, boolean useLikelihood) {
        this(probabilisticGraph, useLikelihood, null);
    }

    public MultivariateFunction fitness() {
        return point -> {

            Map<String, DoubleTensor> values = convertFromPoint(
                point,
                latentVariables,
                latentShapes
            );

            double logOfTotalProbability = useLikelihood ?
                probabilisticGraph.logLikelihood(values) :
                probabilisticGraph.logProb(values);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    public static boolean isValidInitialFitness(double fitnessValue) {
        return fitnessValue == Double.NEGATIVE_INFINITY || Double.isNaN(fitnessValue);
    }

}