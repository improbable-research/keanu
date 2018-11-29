package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex {
    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left  vertex A
     * @param right vertex B
     */

    @ExportVertexToPythonBindings
    public MatrixMultiplicationVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                                      @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(getResultingShape(left.getShape(), right.getShape()),
            left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.matrixMultiply(r);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        PartialDerivatives dOutputsWrtLeft = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                right.getValue(),
                true
            );

        PartialDerivatives dOutputsWrtRight = PartialDerivatives
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputsWithRespectToSelf,
                left.getValue(),
                false
            );

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, dOutputsWrtLeft);
        partials.put(right, dOutputsWrtRight);

        return partials;
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dLeftWrtInputs, PartialDerivatives dRightWrtInputs) {

        // dc = A * db + da * B;
        PartialDerivatives partialsFromLeft;
        PartialDerivatives partialsFromRight;

        if (dLeftWrtInputs.isEmpty()) {
            partialsFromLeft = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromLeft = PartialDerivatives.matrixMultiplyAlongOfDimensions(dLeftWrtInputs, right.getValue(), true);
        }

        if (dRightWrtInputs.isEmpty()) {
            partialsFromRight = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromRight = PartialDerivatives.matrixMultiplyAlongOfDimensions(dRightWrtInputs, left.getValue(), false);
        }

        return partialsFromLeft.add(partialsFromRight);
    }

    private static long[] getResultingShape(long[] left, long[] right) {
        if (left.length != 2 || right.length != 2) {
            throw new IllegalArgumentException("Matrix multiply must be used on matrices");
        }

        if (left[1] != right[0]) {
            throw new IllegalArgumentException("Can not multiply matrices of shapes " + Arrays.toString(left) + " X " + Arrays.toString(right));
        }

        return new long[]{left[0], right[1]};
    }
}
