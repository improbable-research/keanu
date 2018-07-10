package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Arrays;
import java.util.List;

public abstract class Optimizer {

    private static final double FLAT_GRADIENT = 1e-16;

    static double[] currentPoint(List<Vertex<DoubleTensor>> continuousLatentVertices) {
        long totalLatentDimensions = totalNumLatentDimensions(continuousLatentVertices);

        if (totalLatentDimensions > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Greater than " + Integer.MAX_VALUE + " latent dimensions not supported");
        }

        int position = 0;
        double[] point = new double[(int) totalLatentDimensions];

        for (Vertex<DoubleTensor> vertex : continuousLatentVertices) {
            double[] values = vertex.getValue().asFlatDoubleArray();
            System.arraycopy(values, 0, point, position, values.length);
            position += values.length;
        }

        return point;
    }

    static long totalNumLatentDimensions(List<? extends Vertex<DoubleTensor>> continuousLatentVertices) {
        return continuousLatentVertices.stream().mapToLong(FitnessFunction::numDimensions).sum();
    }

    static void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().getAsDouble();
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }
}
