package io.improbable.keanu.algorithms.tensorVariational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;

public class TensorFitnessFunction {

    protected final List<Vertex> probabilisticVertices;
    protected final List<? extends Vertex> latentVertices;
    protected final Map<String, Long> exploreSettingAll;

    public TensorFitnessFunction(List<Vertex> probabilisticVertices, List<? extends Vertex> latentVertices) {
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

    static void setAndCascadePoint(double[] point, List<? extends Vertex> latentVertices, Map<String, Long> exploreSettingAll) {

        int position = 0;
        for (Vertex vertex : latentVertices) {

            int dimensions = numDimensions(vertex);

            if (vertex.getValue() instanceof DoubleTensor) {
                double[] values = new double[dimensions];

                System.arraycopy(point, position, values, 0, values.length);

                Vertex<DoubleTensor> castedVertex = ((Vertex<DoubleTensor>) vertex);
                DoubleTensor newTensor = DoubleTensor.create(values, castedVertex.getValue().getShape());
                castedVertex.setValue(newTensor);
            } else {
                ((Vertex<Double>) vertex).setValue(point[position]);
            }

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