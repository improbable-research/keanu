package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.setAndCascadePoint;

public class FitnessFunction {

    private final List<Vertex> probabilisticVertices;
    private final List<? extends Vertex<DoubleTensor>> latentVertices;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunction(List<Vertex> probabilisticVertices,
                           List<? extends Vertex<DoubleTensor>> latentVertices,
                           BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunction(List<Vertex> probabilisticVertices,
                           List<? extends Vertex<DoubleTensor>> latentVertices) {
        this(probabilisticVertices, latentVertices, null);
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point, latentVertices);
            double logOfTotalProbability = logOfTotalProbability(probabilisticVertices);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    public static double logOfTotalProbability(List<? extends Vertex> probabilisticVertices) {
        double sum = 0.0;
        for (Vertex<?> vertex : probabilisticVertices) {
            sum += vertex.logProbAtValue();
        }

        return sum;
    }

    public static boolean isValidInitialFitness(double fitnessValue) {
        return fitnessValue == Double.NEGATIVE_INFINITY || fitnessValue == Double.NaN;
    }

}