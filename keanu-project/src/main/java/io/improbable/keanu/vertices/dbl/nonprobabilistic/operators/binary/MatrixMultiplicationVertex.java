package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex {
    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left  vertex A
     * @param right vertex B
     */

    public MatrixMultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(getResultingShape(left.getShape(), right.getShape()),
            left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.matrixMultiply(r);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        PartialDerivatives partialsLeft = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                right.getValue(),
                true
            );

        PartialDerivatives partialsRight = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                left.getValue(),
                false
            );

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, partialsLeft);
        partials.put(right, partialsRight);

        return partials;
    }

    @Override
    protected DualNumber dualOp(DualNumber l, DualNumber r) {
        return l.matrixMultiplyBy(r);
    }

    private static int[] getResultingShape(int[] left, int[] right) {
        TensorShapeValidation.checkShapesCanBeMatrixMultiplied(left, right);
        return new int[]{left[0], right[1]};
    }
}
