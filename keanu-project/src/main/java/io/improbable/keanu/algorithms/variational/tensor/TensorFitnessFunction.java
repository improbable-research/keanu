package io.improbable.keanu.algorithms.variational.tensor;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;

public class TensorFitnessFunction {

    private final List<Vertex> probabilisticVertices;
    private final List<? extends Vertex<DoubleTensor>> latentVertices;
    private final Map<Long, Long> exploreSettingAll;

    public TensorFitnessFunction(List<Vertex> probabilisticVertices, List<? extends Vertex<DoubleTensor>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.exploreSettingAll = VertexValuePropagation.exploreSetting(latentVertices);
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point, latentVertices, exploreSettingAll);
            return logOfTotalProbability(probabilisticVertices);
        };
    }

    static void setAndCascadePoint(double[] point, List<? extends Vertex<DoubleTensor>> latentVertices, Map<Long, Long> exploreSettingAll) {

        int position = 0;
        for (Vertex<DoubleTensor> vertex : latentVertices) {

            int dimensions = numDimensions(vertex);

            double[] values = new double[dimensions];
            System.arraycopy(point, position, values, 0, dimensions);

            DoubleTensor newTensor = DoubleTensor.create(values, vertex.getValue().getShape());
            vertex.setValue(newTensor);

            position += dimensions;
        }

        VertexValuePropagation.cascadeUpdate(latentVertices, exploreSettingAll);
    }

    static int numDimensions(Vertex<DoubleTensor> vertex) {
        return vertex.getValue().getLength();
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