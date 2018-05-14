package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class ProbabilisticVertexShaping {

    private ProbabilisticVertexShaping(){
    }

    /**
     * This is a common function to check that a vertex's parent tensors are either
     * the same shape of the vertex in question OR scalar.
     *
     * @param shape   the shape of the vertex being checked
     * @param tensors the tensors of the parent vertices
     */
    public static void checkParentShapes(int[] shape, DoubleTensor... tensors) {

        Set<TensorShape> nonScalarShapes = getNonScalarShapes(tensors);

        if (!nonScalarShapes.isEmpty()) {

            boolean moreThanOneNonScalarShape = nonScalarShapes.size() > 1;
            boolean nonScalarShapeDoesNotMatchProposal = !Arrays.equals(nonScalarShapes.iterator().next().getShape(), shape);

            if (moreThanOneNonScalarShape) {
                throw new IllegalArgumentException("More than a single non-scalar parent shape");
            }

            if (nonScalarShapeDoesNotMatchProposal) {
                throw new IllegalArgumentException("Proposed shape does not match parent shapes");
            }
        }
    }

    /**
     * This implies a vertex's tensor shape based on it's parents tensor shapes.
     *
     * @param tensors the parent tensors of the vertex that needs it's shape implied.
     * @return a suggested shape for the vertex given it's parent vertices shapes.
     */
    public static int[] getShapeProposal(DoubleTensor... tensors) {
        Set<TensorShape> nonScalarShapes = getNonScalarShapes(tensors);

        if (nonScalarShapes.isEmpty()) {
            return new int[]{1, 1};
        } else if (nonScalarShapes.size() == 1) {
            return nonScalarShapes.iterator().next().getShape();
        } else {
            throw new IllegalArgumentException("hyper param shapes must match or be scalar");
        }
    }

    private static Set<TensorShape> getNonScalarShapes(DoubleTensor... tensors) {
        Set<TensorShape> nonScalarShapes = new HashSet<>();
        for (DoubleTensor tensor : tensors) {
            if (!tensor.isScalar()) {
                nonScalarShapes.add(new TensorShape(tensor.getShape()));
            }
        }
        return nonScalarShapes;
    }

    private static class TensorShape {

        private int[] shape;

        public TensorShape(int[] shape) {
            this.shape = shape;
        }

        public int[] getShape() {
            return shape;
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
