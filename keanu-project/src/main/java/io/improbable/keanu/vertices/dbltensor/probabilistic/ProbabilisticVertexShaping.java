package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.TensorShape;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class ProbabilisticVertexShaping {

    public static void checkParentShapes(int[] shape, DoubleTensor... tensors) {

        Set<TensorShape> nonScalarShapes = getNonScalarShapes(tensors);

        if (!nonScalarShapes.isEmpty()) {

            boolean moreThanOneNonScalarShape = nonScalarShapes.size() > 1;
            boolean nonScalarShapeDoesNotMatchProposal = !Arrays.equals(nonScalarShapes.iterator().next().getShape(), shape);

            if (moreThanOneNonScalarShape || nonScalarShapeDoesNotMatchProposal) {
                throw new IllegalArgumentException("invalid hyper parameter shapes");
            }
        }
    }

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
}
