package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FitnessFunction {

    protected final List<Vertex<?>> probabilisticVertices;
    protected final List<? extends Vertex<Double>> latentVertices;
    protected final Map<String, Map<String, Integer>> setAndCascadeCache;

    public FitnessFunction(List<Vertex<?>> probabilisticVertices, List<? extends Vertex<Double>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.setAndCascadeCache = new HashMap<>();
    }

    public MultivariateFunction fitness() {
        return point -> {
            setPoint(point);
            return logOfTotalProbability();
        };
    }

    protected void setPoint(double[] point) {
        for (int i = 0; i < point.length; i++) {
            Vertex<Double> v = latentVertices.get(i);

            Map<String, Integer> explored = setAndCascadeCache.computeIfAbsent(v.getId(), (id) -> v.exploreSetting());

            v.setAndCascade(point[i], explored);
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