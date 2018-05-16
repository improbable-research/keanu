package io.improbable.keanu.vertices.dbltensor;

import java.util.Arrays;
import java.util.Set;

import static java.util.stream.Collectors.toSet;

public class TensorShapeValidation {

    private TensorShapeValidation() {
    }

    /**
     * This is a common function to check that tensors are either
     * the same shape of the proposal in question OR scalar.
     *
     * @param proposalShape the tensor shape being validated
     * @param tensors       the tensors being validated against
     * @throws IllegalArgumentException if there is more than one non-scalar shape OR if the non-scalar shape does
     *                                  not match the proposal shape.
     */
    public static void checkTensorsMatchNonScalarShapeOrAreScalar(int[] proposalShape, Tensor... tensors) {

        Set<TensorShape> nonScalarShapes = getNonScalarShapes(tensors);

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
     * @param tensors the tensors for shape checking
     * @return either a scalar shape OR the single non-scalar shape.
     * @throws IllegalArgumentException if there is more than one non-scalar shape
     */
    public static int[] checkHasSingleNonScalarShapeOrAllScalar(DoubleTensor... tensors) {
        Set<TensorShape> nonScalarShapes = getNonScalarShapes(tensors);

        if (nonScalarShapes.isEmpty()) {
            return Tensor.SCALAR_SHAPE;
        } else if (nonScalarShapes.size() == 1) {
            return nonScalarShapes.iterator().next().getShape();
        } else {
            throw new IllegalArgumentException("Shapes must match or be scalar");
        }
    }

    private static Set<TensorShape> getNonScalarShapes(Tensor... tensors) {
        return Arrays.stream(tensors)
            .filter(shape -> !shape.isScalar())
            .map(tensor -> new TensorShape(tensor.getShape(), tensor.getLength()))
            .collect(toSet());
    }

    private static class TensorShape {

        private int[] shape;
        private int length;

        public TensorShape(int[] shape, int length) {
            this.shape = shape;
            this.length = length;
        }

        public int[] getShape() {
            return shape;
        }

        public int getLength() {
            return length;
        }

        public boolean isScalar() {
            return length == 1;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            TensorShape that = (TensorShape) o;

            return Arrays.equals(shape, that.shape);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(shape);
        }
    }
}
