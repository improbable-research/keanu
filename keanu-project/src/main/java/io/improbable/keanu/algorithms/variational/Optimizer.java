package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;

public abstract class Optimizer {

    protected double[] currentPoint(List<Vertex<DoubleTensor>> continuousLatentVertices) {
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

    protected long totalNumLatentDimensions(List<? extends Vertex<DoubleTensor>> continuousLatentVertices) {
        return continuousLatentVertices.stream().mapToLong(FitnessFunction::numDimensions).sum();
    }
}
