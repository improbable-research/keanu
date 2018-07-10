package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.function.BiConsumer;

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

    static void setAndCascadePoint(double[] point, List<? extends Vertex<DoubleTensor>> latentVertices) {

        int position = 0;
        for (Vertex<DoubleTensor> vertex : latentVertices) {

            int dimensions = (int) numDimensions(vertex);

            double[] values = new double[dimensions];
            System.arraycopy(point, position, values, 0, dimensions);

            DoubleTensor newTensor = DoubleTensor.create(values, vertex.getValue().getShape());
            vertex.setValue(newTensor);

            position += dimensions;
        }

        VertexValuePropagation.cascadeUpdate(latentVertices);
    }

    static long numDimensions(Vertex<DoubleTensor> vertex) {
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