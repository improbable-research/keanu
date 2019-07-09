package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.getMatrixMultiplicationResultingShape;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex implements Differentiable {

    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left  vertex A
     * @param right vertex B
     */
    @ExportVertexToPythonBindings
    public MatrixMultiplicationVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                                      @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(getMatrixMultiplicationResultingShape(left.getShape(), right.getShape()),
            left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.matrixMultiply(r);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        PartialDerivative dOutputsWrtLeft = PartialDerivative
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputWithRespectToSelf,
                right.getValue(),
                true
            );

        PartialDerivative dOutputsWrtRight = PartialDerivative
            .matrixMultiplyAlongWrtDimensions(
                derivativeOfOutputWithRespectToSelf,
                left.getValue(),
                false
            );

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(left, dOutputsWrtLeft);
        partials.put(right, dOutputsWrtRight);

        return partials;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
        PartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);

        // dc = A * db + da * B;
        PartialDerivative partialsFromLeft = PartialDerivative.matrixMultiplyAlongOfDimensions(
            dLeftWrtInput,
            right.getValue(),
            true
        );

        PartialDerivative partialsFromRight = PartialDerivative.matrixMultiplyAlongOfDimensions(
            dRightWrtInput,
            left.getValue(),
            false
        );

        return partialsFromLeft.add(partialsFromRight);
    }
}
