package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;

public class FitnessFunction {

    protected final List<Vertex<?>> probabilisticVertices;
    protected final List<? extends Vertex<Double>> latentVertices;
    protected final Map<String, Long> exploreSettingAll;

    public FitnessFunction(List<Vertex<?>> probabilisticVertices, List<? extends Vertex<Double>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.exploreSettingAll = VertexValuePropagation.exploreSetting(latentVertices);
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point);
            return logOfTotalProbability();
        };
    }

    protected void setAndCascadePoint(double[] point) {
        for (int i = 0; i < point.length; i++) {
            Vertex<Double> vertex = latentVertices.get(i);
            vertex.setValue(point[i]);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices, exploreSettingAll);
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