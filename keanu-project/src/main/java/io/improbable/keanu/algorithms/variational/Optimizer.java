package io.improbable.keanu.algorithms.variational;

import java.util.Collection;
import java.util.List;
import java.util.function.BiConsumer;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.Vertex;

public interface Optimizer {

    void onFitnessCalculation(BiConsumer<double[], Double> fitnessCalculationHandler);

    double maxAPosteriori();

    double maxLikelihood();

    BayesianNetwork getBayesianNetwork();

    static Optimizer of(BayesianNetwork network) {
        if (network.getDiscreteLatentVertices().isEmpty()) {
            return GradientOptimizer.of(network);
        } else {
            return NonGradientOptimizer.of(network);
        }
    }

    static Optimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    static double[] currentPoint(List<? extends Vertex<? extends NumberTensor>> continuousLatentVertices) {
        long totalLatentDimensions = totalNumberOfLatentDimensions(continuousLatentVertices);

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (Vertex<? extends NumberTensor> vertex : continuousLatentVertices) {
            double[] values = vertex.getValue().asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    static long totalNumberOfLatentDimensions(List<? extends Vertex<? extends NumberTensor>> continuousLatentVertices) {
        return continuousLatentVertices.stream().mapToLong(FitnessFunction::numDimensions).sum();
    }
}
