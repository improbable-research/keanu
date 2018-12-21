package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.optim.SimpleBounds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class creates an Apache math commons simple bounds object for a given collection of
 * vertices and their staring point. This can be used for some of the Apache math commons
 * optimizer, namely the BOBYQAOptimizer.
 */
@RequiredArgsConstructor
class ApacheMathSimpleBoundsCalculator {

    private final double boundsRange;
    private final OptimizerBounds optimizerBounds;

    SimpleBounds getBounds(List<? extends Variable> latentVariables, double[] startPoint) {
        List<Double> minBounds = new ArrayList<>();
        List<Double> maxBounds = new ArrayList<>();

        for (Variable variable : latentVariables) {

            if (optimizerBounds.hasBound(variable.getReference())) {
                validateBoundsForVariable(variable);
                addBoundsForVariable(variable, minBounds, maxBounds);
            } else {
                int length = TensorShape.getLengthAsInt(variable.getShape());
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

    private void addBoundsForVariable(Variable variable,
                                      List<Double> minBounds,
                                      List<Double> maxBounds) {

        DoubleTensor lowerBound = optimizerBounds.getLower(variable.getReference());
        DoubleTensor upperBound = optimizerBounds.getUpper(variable.getReference());

        if (lowerBound.isScalar()) {
            minBounds.addAll(DoubleTensor.create(lowerBound.scalar(), variable.getShape()).asFlatList());
        } else {
            minBounds.addAll(lowerBound.asFlatList());
        }

        if (upperBound.isScalar()) {
            maxBounds.addAll(DoubleTensor.create(upperBound.scalar(), variable.getShape()).asFlatList());
        } else {
            maxBounds.addAll(upperBound.asFlatList());
        }
    }

    private void validateBoundsForVariable(Variable variable) {
        if (!optimizerBounds.getLower(variable.getReference()).isScalar() && !Arrays.equals(variable.getShape(), optimizerBounds.getLower(variable.getReference()).getShape())) {
            throw new IllegalArgumentException("Lower bounds shape does not match variable shape");
        }
        if (!optimizerBounds.getUpper(variable.getReference()).isScalar() && !Arrays.equals(variable.getShape(), optimizerBounds.getUpper(variable.getReference()).getShape())) {
            throw new IllegalArgumentException("Upper bounds shape does not match variable shape");
        }
    }

}
