package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.optim.SimpleBounds;

/**
 * This class creates an Apache math commons simple bounds object for a given collection of vertices
 * and their staring point. This can be used for some of the Apache math commons optimizer, namely
 * the BOBYQAOptimizer.
 */
@RequiredArgsConstructor
class ApacheMathSimpleBoundsCalculator {

    private final double boundsRange;
    private final OptimizerBounds optimizerBounds;

    SimpleBounds getBounds(
            List<? extends Vertex<DoubleTensor>> latentVertices, double[] startPoint) {
        List<Double> minBounds = new ArrayList<>();
        List<Double> maxBounds = new ArrayList<>();

        for (Vertex<DoubleTensor> vertex : latentVertices) {

            if (optimizerBounds.hasBound(vertex)) {
                validateBoundsForVertex(vertex);
                addBoundsForVertex(vertex, minBounds, maxBounds);
            } else {
                int length = (int) TensorShape.getLength(vertex.getShape());
                int startIndex = minBounds.size();
                for (int i = 0; i < length; i++) {
                    minBounds.add(startPoint[i + startIndex] - boundsRange);
                    maxBounds.add(startPoint[i + startIndex] + boundsRange);
                }
            }
        }

        return new SimpleBounds(
                minBounds.stream().mapToDouble(d -> d).toArray(),
                maxBounds.stream().mapToDouble(d -> d).toArray());
    }

    private void addBoundsForVertex(
            Vertex<DoubleTensor> vertex, List<Double> minBounds, List<Double> maxBounds) {

        DoubleTensor lowerBound = optimizerBounds.getLower(vertex);
        DoubleTensor upperBound = optimizerBounds.getUpper(vertex);

        if (lowerBound.isScalar()) {
            minBounds.addAll(
                    DoubleTensor.create(lowerBound.scalar(), vertex.getShape()).asFlatList());
        } else {
            minBounds.addAll(lowerBound.asFlatList());
        }

        if (upperBound.isScalar()) {
            maxBounds.addAll(
                    DoubleTensor.create(upperBound.scalar(), vertex.getShape()).asFlatList());
        } else {
            maxBounds.addAll(upperBound.asFlatList());
        }
    }

    private void validateBoundsForVertex(Vertex<DoubleTensor> vertex) {
        int[] vertexShape = vertex.getShape();
        if (!optimizerBounds.getLower(vertex).isScalar()
                && !Arrays.equals(vertexShape, optimizerBounds.getLower(vertex).getShape())) {
            throw new IllegalArgumentException("Lower bounds shape does not match vertex shape");
        }
        if (!optimizerBounds.getUpper(vertex).isScalar()
                && !Arrays.equals(vertexShape, optimizerBounds.getUpper(vertex).getShape())) {
            throw new IllegalArgumentException("Upper bounds shape does not match vertex shape");
        }
    }
}
