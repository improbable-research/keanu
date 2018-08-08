package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.setAndCascadePoint;

import java.util.List;
import java.util.function.BiConsumer;

import org.apache.commons.math3.analysis.MultivariateFunction;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
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
            double logOfTotalProbability = logOfTotalProbability(outputVertices);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    public static double logOfTotalProbability(List<? extends Vertex> vertices) {
        for (Vertex<?> v : vertices) {
            if (!v.isProbabilistic() && v.isObserved() && !v.matchesObservation()) {
                return Double.NEGATIVE_INFINITY;
            }
        }
        double sum = 0.0;
        for (Probabilistic vertex : Probabilistic.keepOnlyProbabilisticVertices(vertices)) {
            sum += vertex.logProbAtValue();
        }

        return sum;
    }

    public static boolean isValidInitialFitness(double fitnessValue) {
        return fitnessValue == Double.NEGATIVE_INFINITY || fitnessValue == Double.NaN;
    }

}