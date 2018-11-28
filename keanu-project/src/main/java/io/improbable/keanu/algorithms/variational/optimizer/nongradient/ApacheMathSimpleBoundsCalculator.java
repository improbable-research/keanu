package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.optim.SimpleBounds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

/**
 * This class creates an Apache math commons simple bounds object for a given collection of
 * vertices and their staring point. This can be used for some of the Apache math commons
 * optimizer, namely the BOBYQAOptimizer.
 */
@RequiredArgsConstructor
class ApacheMathSimpleBoundsCalculator {

    private final double boundsRange;
    private final OptimizerBounds optimizerBounds;

    SimpleBounds getBounds(List<? extends Vertex> latentVertices, double[] startPoint) {

        List<String> latentVertexNames = latentVertices.stream()
            .map(Vertex::getUniqueStringReference)
            .collect(toList());

        Map<String, long[]> latentVertexShapes = latentVertices.stream()
            .collect(toMap(
                Vertex::getUniqueStringReference,
                Vertex::getShape
            ));

        return getBounds(latentVertexNames, latentVertexShapes, startPoint);
    }

    SimpleBounds getBounds(List<String> latentVariables, Map<String, long[]> latentShapes, double[] startPoint) {
        List<Double> minBounds = new ArrayList<>();
        List<Double> maxBounds = new ArrayList<>();

        for (String variable : latentVariables) {

            long[] variableShape = latentShapes.get(variable);
            if (optimizerBounds.hasBound(variable)) {
                validateBoundsForVariable(variable, variableShape);
                addBoundsForVariable(variable, variableShape, minBounds, maxBounds);
            } else {
                int length = TensorShape.getLengthAsInt(variableShape);
                int startIndex = minBounds.size();
                for (int i = 0; i < length; i++) {
                    minBounds.add(startPoint[i + startIndex] - boundsRange);
                    maxBounds.add(startPoint[i + startIndex] + boundsRange);
                }
            }
        }

        return new SimpleBounds(
            minBounds.stream().mapToDouble(d -> d).toArray(),
            maxBounds.stream().mapToDouble(d -> d).toArray()
        );
    }

    private void addBoundsForVariable(String variable,
                                      long[] variableShape,
                                      List<Double> minBounds,
                                      List<Double> maxBounds) {

        DoubleTensor lowerBound = optimizerBounds.getLower(variable);
        DoubleTensor upperBound = optimizerBounds.getUpper(variable);

        if (lowerBound.isScalar()) {
            minBounds.addAll(DoubleTensor.create(lowerBound.scalar(), variableShape).asFlatList());
        } else {
            minBounds.addAll(lowerBound.asFlatList());
        }

        if (upperBound.isScalar()) {
            maxBounds.addAll(DoubleTensor.create(upperBound.scalar(), variableShape).asFlatList());
        } else {
            maxBounds.addAll(upperBound.asFlatList());
        }
    }

    private void validateBoundsForVariable(String variable, long[] variableShape) {
        if (!optimizerBounds.getLower(variable).isScalar() && !Arrays.equals(variableShape, optimizerBounds.getLower(variable).getShape())) {
            throw new IllegalArgumentException("Lower bounds shape does not match variable shape");
        }
        if (!optimizerBounds.getUpper(variable).isScalar() && !Arrays.equals(variableShape, optimizerBounds.getUpper(variable).getShape())) {
            throw new IllegalArgumentException("Upper bounds shape does not match variable shape");
        }
    }

}
