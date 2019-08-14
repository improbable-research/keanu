package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;
import static org.apache.commons.lang3.ArrayUtils.removeAll;

public class TensorShapeValidation {

    private TensorShapeValidation() {
    }

    /**
     * This is a common function to check that tensors are either
     * the same shape of the proposal in question OR length one.
     *
     * @param proposalShape the tensor shape being validated
     * @param shapes        the tensors being validated against
     * @throws IllegalArgumentException if there is more than one non length one shape OR if the non length one shape does
     *                                  not match the proposal shape.
     */
    public static void checkTensorsMatchNonLengthOneShapeOrAreLengthOne(long[] proposalShape, long[]... shapes) {

        Set<TensorShape> nonLengthOneShapes = getNonLengthOneShapes(shapes);

        if (!nonLengthOneShapes.isEmpty()) {

            boolean moreThanOneNonLengthOneShape = nonLengthOneShapes.size() > 1;

            if (moreThanOneNonLengthOneShape) {
                throw new IllegalArgumentException("More than a single non length one shape");
            }

            long[] nonLengthOneShape = nonLengthOneShapes.iterator().next().getShape();
            boolean nonLengthOneShapeDoesNotMatchProposal = !Arrays.equals(nonLengthOneShape, proposalShape);

            if (nonLengthOneShapeDoesNotMatchProposal) {
                throw new IllegalArgumentException(
                    "Proposed shape "
                        + Arrays.toString(proposalShape)
                        + " does not match other non length one shapes "
                        + Arrays.toString(nonLengthOneShape)
                );
            }
        }
    }

    /**
     * Check if the given dimension exists within the shape
     *
     * @param dimension Proposed dimension
     * @param shape     Shape to check
     * @throws IllegalArgumentException if the dimension exceeds the rank of the shape
     */
    public static void checkDimensionExistsInShape(int dimension, long[] shape) {
        if (dimension >= shape.length) {
            throw new IllegalArgumentException(String.format("Dimension %d does not exist in tensor of rank %d", dimension, shape.length));
        }
    }

    public static void checkTensorsAreScalar(String message, long[]... shapes) {
        Set<TensorShape> nonScalarShapes = getNonScalarShapes(shapes);

        if (!nonScalarShapes.isEmpty()) {
            throw new IllegalArgumentException(message);
        }
    }

    /**
     * This ensures there is at most a single non length one shape.
     *
     * @param shapes the tensors for shape checking
     * @return either a length one shape OR the single non length one shape.
     * @throws IllegalArgumentException if there is more than one non length one shape or multiple ranks of length 1 shapes
     */
    public static long[] checkHasOneNonLengthOneShapeOrAllLengthOne(long[]... shapes) {
        Set<TensorShape> nonLengthOneShapes = getNonLengthOneShapes(shapes);
        List<TensorShape> lengthOneShapes = getLengthOneShapesSortedByRank(shapes);

        if (nonLengthOneShapes.isEmpty()) {
            if (!lengthOneShapes.isEmpty()) {
                return lengthOneShapes.get(0).getShape();
            }
        } else if (nonLengthOneShapes.size() == 1) {
            return nonLengthOneShapes.iterator().next().getShape();
        }

        throw new IllegalArgumentException("Shapes must match or be length one but were: " +
            Arrays.stream(shapes)
                .map(Arrays::toString)
                .collect(Collectors.joining(","))
        );
    }

    public static boolean isBroadcastable(long[] left, long[] right) {
        try {
            TensorShape.getBroadcastResultShape(left, right);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /**
     * @param predicate shape of predicate
     * @param thn       shape of then
     * @param els       shape of else
     * @return the result shape of predicate ? thn : els. This will be the shape of the predicate if it is not a length
     * one shape or the shape of thn/els if the predicate is length one and thn/else are not.
     */
    public static long[] checkTernaryConditionShapeIsValid(long[] predicate, long[] thn, long[] els) {
        Preconditions.checkArgument(Arrays.equals(thn, els),
            "Then shape " + Arrays.toString(thn) +
                " must match else shape " + Arrays.toString(els)
        );
        return checkHasOneNonLengthOneShapeOrAllLengthOne(predicate, thn, els);
    }

    public static void checkShapeIsSquareMatrix(long[] shape) {
        if (shape.length < 2) {
            throw new IllegalArgumentException("Input tensor must be a matrix");
        }

        if (shape[shape.length - 1] != shape[shape.length - 2]) {
            throw new IllegalArgumentException("Input matrix must be square");
        }
    }

    private static Set<TensorShape> getNonLengthOneShapes(long[]... shapes) {
        return Arrays.stream(shapes)
            .map(TensorShape::new)
            .filter(shape -> !shape.isLengthOne())
            .collect(toSet());
    }

    /**
     * @param shapes shapes to check
     * @return a list of all length one shapes sorted from longest to shortest
     * length one shapes. E.g. if given ([1,1], [1,1,1], [2,3]) then ([1,1,1], [1,1])
     * will be returned.
     */
    private static List<TensorShape> getLengthOneShapesSortedByRank(long[]... shapes) {
        return Arrays.stream(shapes)
            .map(TensorShape::new)
            .filter(TensorShape::isLengthOne)
            .sorted(Comparator.comparingInt(s -> ((TensorShape) s).getRank()).reversed())
            .collect(toList());
    }

    private static Set<TensorShape> getNonScalarShapes(long[]... shapes) {
        return Arrays.stream(shapes)
            .map(TensorShape::new)
            .filter(shape -> !shape.isScalar())
            .collect(toSet());
    }

    public static void checkShapesMatch(long[] actual, long[] expected) {
        if (!Arrays.equals(actual, expected)) {
            throw new IllegalArgumentException(String.format("Expected shape %s but got %s", Arrays.toString(expected), Arrays.toString(actual)));
        }
    }

    public static long[] checkAllShapesMatch(long[]... shapes) {
        return checkAllShapesMatch(Arrays.stream(shapes), Optional.empty());
    }

    public static long[] checkAllShapesMatch(String errorMessage, long[]... shapes) {
        return checkAllShapesMatch(Arrays.stream(shapes), Optional.of(errorMessage));
    }

    public static long[] checkAllShapesMatch(String errorMessage, Collection<long[]> shapes) {
        return checkAllShapesMatch(shapes.stream(), Optional.of(errorMessage));
    }

    public static long[] checkAllShapesMatch(Collection<long[]> shapes) {
        return checkAllShapesMatch(shapes.stream(), Optional.empty());
    }

    private static long[] checkAllShapesMatch(Stream<long[]> shapesStream, Optional<String> errorMessage) {
        Set<TensorShape> uniqueShapes = shapesStream
            .map(TensorShape::new)
            .collect(toSet());

        if (uniqueShapes.size() != 1) {
            throw new IllegalArgumentException(errorMessage.orElse("Shapes must match"));
        }

        return uniqueShapes.iterator().next().getShape();
    }

    public static long[] checkShapesCanBeConcatenated(int dimension, long[]... shapes) {
        long[] concatShape = Arrays.copyOf(shapes[0], shapes[0].length);

        for (int i = 1; i < shapes.length; i++) {
            int rank = shapes[i].length;

            if (rank <= dimension) {
                throw new IllegalArgumentException(String.format("Cannot concat operand %d because dimension %d is greater than or equal to its rank %d", i, dimension, rank));
            }

            if (rank != concatShape.length) {
                throw new IllegalArgumentException("Cannot concat shapes of different ranks");
            }

            for (int dim = 0; dim < rank; dim++) {
                if (dim == dimension) {
                    concatShape[dim] += shapes[i][dim];
                } else {
                    if (shapes[i][dim] != concatShape[dim]) {
                        throw new IllegalArgumentException("Cannot concat mismatched shapes");
                    }
                }
            }
        }
        return concatShape;
    }

    public static void checkIndexIsValid(long[] shape, long... index) {
        if (shape.length != index.length) {
            throw new IllegalArgumentException(
                "Length of desired index " + Arrays.toString(index) + " must match the length of the shape " + Arrays.toString(shape));
        }

        for (int i = 0; i < index.length; i++) {

            if (index[i] >= shape[i]) {
                throw new IllegalArgumentException(
                    "Invalid index " + Arrays.toString(index) + " for shape " + Arrays.toString(shape)
                );
            }

        }
    }

    public static long[] getTensorMultiplyResultShape(long[] leftShape, long[] rightShape, int[] dimsLeft, int[] dimsRight) {

        if (dimsLeft.length != dimsRight.length) {
            throw new IllegalArgumentException("Tensor multiply must match dimension lengths " +
                toStringArgs(leftShape, rightShape, dimsLeft, dimsRight)
            );
        }


        for (int i = 0; i < dimsLeft.length; i++) {

            if (dimsLeft[i] >= leftShape.length || dimsLeft[i] < 0) {
                throw new IllegalArgumentException("Left dimensions " + Arrays.toString(dimsLeft) +
                    " is invalid for left shape " + Arrays.toString(leftShape)
                );
            }

            if (dimsRight[i] >= rightShape.length || dimsRight[i] < 0) {
                throw new IllegalArgumentException("Right dimensions " + Arrays.toString(dimsRight) +
                    " is invalid for right shape " + Arrays.toString(rightShape)
                );
            }

            if (leftShape[dimsLeft[i]] != rightShape[dimsRight[i]]) {
                throw new IllegalArgumentException("Cannot tensor multiply dimension " + i + " for " +
                    toStringArgs(leftShape, rightShape, dimsLeft, dimsRight)
                );
            }
        }

        return TensorShape.concat(removeAll(leftShape, dimsLeft), removeAll(rightShape, dimsRight));
    }

    private static String toStringArgs(long[] leftShape, long[] rightShape, int[] dimsLeft, int[] dimsRight) {
        return "left shape: " + Arrays.toString(leftShape) + " right shape: " + Arrays.toString(rightShape) + " on left dimensions " +
            Arrays.toString(dimsLeft) + " and right dimensions " + Arrays.toString(dimsRight);
    }

    public static long[] getMatrixMultiplicationResultingShape(long[] left, long[] right) {
        if (left.length < 2 || right.length < 2) {
            throw new IllegalArgumentException("Cannot matrix multiply with shapes " + Arrays.toString(left) + " and " + Arrays.toString(right));
        }

        if (left[left.length - 1] != right[right.length - 2]) {
            throw new IllegalArgumentException("Cannot matrix multiply with shapes " + Arrays.toString(left) + " and " + Arrays.toString(right));
        }

        if (left.length == 2 && right.length == 2) {
            return new long[]{left[0], right[1]};
        } else {
            final long[] leftBatchShape = ArrayUtils.subarray(left, 0, left.length - 2);
            final long[] rightBatchShape = ArrayUtils.subarray(right, 0, right.length - 2);
            final long[] batchShape = TensorShape.getBroadcastResultShape(leftBatchShape, rightBatchShape);
            return TensorShape.concat(batchShape, new long[]{left[left.length - 1], right[right.length - 2]});
        }
    }
}
