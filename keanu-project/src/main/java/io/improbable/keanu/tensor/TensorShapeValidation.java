package io.improbable.keanu.tensor;

import java.util.Arrays;
import java.util.Collection;
import java.util.Set;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toSet;

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
        return checkAllShapesMatch(Arrays.stream(shapes));
    }

    public static int[] checkAllShapesMatch(Collection<int[]> shapes) {
        return checkAllShapesMatch(shapes.stream());
    }

    private static int[] checkAllShapesMatch(Stream<int[]> shapesStream) {
        Set<TensorShape> uniqueShapes = shapesStream
            .map(TensorShape::new)
            .collect(toSet());

        if (uniqueShapes.size() != 1) {
            throw new IllegalArgumentException("Shapes must match");
        }

        return uniqueShapes.iterator().next().getShape();
    }

}
