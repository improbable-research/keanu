package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.setAndCascadePoint;

import java.util.List;
import java.util.function.BiConsumer;

import org.apache.commons.math3.analysis.MultivariateFunction;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

public class FitnessFunction {

    private final List<Vertex> outputVertices;
    private final List<? extends Vertex<DoubleTensor>> latentVertices;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunction(List<Vertex> outputVertices,
                           List<? extends Vertex<DoubleTensor>> latentVertices,
                           BiConsumer<double[], Double> onFitnessCalculation) {
        this.outputVertices = outputVertices;
        this.latentVertices = latentVertices;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunction(List<Vertex> outputVertices,
                           List<? extends Vertex<DoubleTensor>> latentVertices) {
        this(outputVertices, latentVertices, null);
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point, latentVertices);
            double logOfTotalProbability = ProbabilityCalculator.calculateLogProbFor(outputVertices);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    public static boolean isValidInitialFitness(double fitnessValue) {
        return fitnessValue == Double.NEGATIVE_INFINITY || fitnessValue == Double.NaN;
    }

}