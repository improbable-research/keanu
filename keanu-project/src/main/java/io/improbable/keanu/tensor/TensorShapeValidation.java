package io.improbable.keanu.tensor;

import static java.util.stream.Collectors.toSet;

import java.util.Arrays;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

import com.google.common.base.Preconditions;

public class TensorShapeValidation {

    private TensorShapeValidation() {
    }

    /**
     * This is a common function to check that tensors are either
     * the same shape of the proposal in question OR scalar.
     *
     * @param proposalShape the tensor shape being validated
     * @param shapes        the tensors being validated against
     * @throws IllegalArgumentException if there is more than one non-scalar shape OR if the non-scalar shape does
     *                                  not match the proposal shape.
     */
    public static void checkTensorsMatchNonScalarShapeOrAreScalar(long[] proposalShape, long[]... shapes) {

        Set<TensorShape> nonScalarShapes = getNonScalarShapes(shapes);

        if (!nonScalarShapes.isEmpty()) {

            boolean moreThanOneNonScalarShape = nonScalarShapes.size() > 1;

            if (moreThanOneNonScalarShape) {
                throw new IllegalArgumentException("More than a single non-scalar shape");
            }

            long[] nonScalarShape = nonScalarShapes.iterator().next().getShape();
            boolean nonScalarShapeDoesNotMatchProposal = !Arrays.equals(nonScalarShape, proposalShape);

            if (nonScalarShapeDoesNotMatchProposal) {
                throw new IllegalArgumentException(
                    "Proposed shape " + Arrays.toString(proposalShape) + " does not match other non scalar shapes"
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

    /**
     * This ensures there is at most a single non-scalar shape.
     *
     * @param shapes the tensors for shape checking
     * @return either a scalar shape OR the single non-scalar shape.
     * @throws IllegalArgumentException if there is more than one non-scalar shape
     */
    public static long[] checkHasSingleNonScalarShapeOrAllScalar(long[]... shapes) {
        Set<TensorShape> nonScalarShapes = getNonScalarShapes(shapes);

        if (nonScalarShapes.isEmpty()) {
            return Tensor.SCALAR_SHAPE;
        } else if (nonScalarShapes.size() == 1) {
            return nonScalarShapes.iterator().next().getShape();
        } else {
            throw new IllegalArgumentException("Shapes must match or be scalar");
        }
    }

    public static void checkShapeIsSquareMatrix(long[] shape) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Input tensor must be a matrix");
        }

        if (shape[0] != shape[1]) {
            throw new IllegalArgumentException("Input matrix must be square");
        }
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
            if (shapes[i].length != concatShape.length) {
                throw new IllegalArgumentException("Cannot concat shapes of different ranks");
            }

            for (int dim = 0; dim < shapes[i].length; dim++) {
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

    public static void checkRankIsAtLeastTwo(long[] shape) {
        Preconditions.checkArgument(shape.length > 1, "Tensors must have rank >=2 : " + Arrays.toString(shape));

    }
}
