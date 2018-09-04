package io.improbable.keanu.tensor;

import static java.util.stream.Collectors.toSet;

import java.util.Arrays;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

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
    public static void checkTensorsMatchNonScalarShapeOrAreScalar(int[] proposalShape, int[]... shapes) {

        Set<TensorShape> nonScalarShapes = getNonScalarShapes(shapes);

        if (!nonScalarShapes.isEmpty()) {

            boolean moreThanOneNonScalarShape = nonScalarShapes.size() > 1;

            if (moreThanOneNonScalarShape) {
                throw new IllegalArgumentException("More than a single non-scalar shape");
            }

            int[] nonScalarShape = nonScalarShapes.iterator().next().getShape();
            boolean nonScalarShapeDoesNotMatchProposal = !Arrays.equals(nonScalarShape, proposalShape);

            if (nonScalarShapeDoesNotMatchProposal) {
                throw new IllegalArgumentException(
                    "Proposed shape " + Arrays.toString(proposalShape) + " does not match other non scalar shapes"
                );
            }
        }
    }

    /**
     * This ensures there is at most a single non-scalar shape.
     *
     * @param shapes the tensors for shape checking
     * @return either a scalar shape OR the single non-scalar shape.
     * @throws IllegalArgumentException if there is more than one non-scalar shape
     */
    public static int[] checkHasSingleNonScalarShapeOrAllScalar(int[]... shapes) {
        Set<TensorShape> nonScalarShapes = getNonScalarShapes(shapes);

        if (nonScalarShapes.isEmpty()) {
            return Tensor.SCALAR_SHAPE;
        } else if (nonScalarShapes.size() == 1) {
            return nonScalarShapes.iterator().next().getShape();
        } else {
            throw new IllegalArgumentException("Shapes must match or be scalar");
        }
    }

    private static Set<TensorShape> getNonScalarShapes(int[]... shapes) {
        return Arrays.stream(shapes)
            .map(TensorShape::new)
            .filter(shape -> !shape.isScalar())
            .collect(toSet());
    }

    public static int[] checkAllShapesMatch(int[]... shapes) {
        return checkAllShapesMatch(Arrays.stream(shapes), Optional.empty());
    }

    public static int[] checkAllShapesMatch(String errorMessage, int[]... shapes) {
        return checkAllShapesMatch(Arrays.stream(shapes), Optional.of(errorMessage));
    }

    public static int[] checkAllShapesMatch(String errorMessage, Collection<int[]> shapes) {
        return checkAllShapesMatch(shapes.stream(), Optional.of(errorMessage));
    }

    public static int[] checkAllShapesMatch(Collection<int[]> shapes) {
        return checkAllShapesMatch(shapes.stream(), Optional.empty());
    }

    private static int[] checkAllShapesMatch(Stream<int[]> shapesStream, Optional<String> errorMessage) {
        Set<TensorShape> uniqueShapes = shapesStream
            .map(TensorShape::new)
            .collect(toSet());

        if (uniqueShapes.size() != 1) {
            throw new IllegalArgumentException(errorMessage.orElse("Shapes must match"));
        }

        return uniqueShapes.iterator().next().getShape();
    }

    public static int[] checkShapesCanBeConcatenated(int dimension, int[]... shapes) {
        int[] concatShape = Arrays.copyOf(shapes[0], shapes[0].length);

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

    public static void checkIndexIsValid(int[] shape, int... index) {
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


}
