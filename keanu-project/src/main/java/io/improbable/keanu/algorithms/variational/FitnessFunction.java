package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;

public class FitnessFunction {

    protected final List<Vertex<?>> probabilisticVertices;
    protected final List<? extends Vertex<Double>> latentVertices;

    public FitnessFunction(List<Vertex<?>> probabilisticVertices, List<? extends Vertex<Double>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point);
            return logOfTotalProbability();
        };
    }

    protected void setAndCascadePoint(double[] point) {
        for (int i = 0; i < point.length; i++) {
            latentVertices
                    .get(i)
                    .setAndCascade(point[i]);
        }
    }

    protected double logOfTotalProbability() {
        double sum = 0.0;
        for (Vertex<?> vertex : probabilisticVertices) {
            sum += vertex.logDensityAtValue();
        }

        return sum;
    }

    public static boolean isValidInitialFitness(double fitnessValue) {
        return fitnessValue == Double.NEGATIVE_INFINITY || fitnessValue == Double.NaN;
    }

}