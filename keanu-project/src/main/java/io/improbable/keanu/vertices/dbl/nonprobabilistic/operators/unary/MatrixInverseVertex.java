package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class MatrixInverseVertex extends DoubleUnaryOpVertex implements Differentiable {

    @ExportVertexToPythonBindings
    public MatrixInverseVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(checkInputIsSquareMatrix(inputVertex.getShape()), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.matrixInverse();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        //dc = -A^-1 * da * A^-1
        DoubleTensor negatedValue = this.getValue().unaryMinus();
        PartialDerivative partial = PartialDerivative.matrixMultiplyAlongOfDimensions(derivativeOfParentWithRespectToInputs, negatedValue, false);
        partial = PartialDerivative.matrixMultiplyAlongOfDimensions(partial, this.getValue(), true);
        return partial;
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor parentValue = getValue();
        DoubleTensor negativeValue = getValue().unaryMinus();

        PartialDerivative newPartials =
            PartialDerivative.matrixMultiplyAlongWrtDimensions(derivativeOfOutputWithRespectToSelf, negativeValue, false);
        newPartials = PartialDerivative.matrixMultiplyAlongWrtDimensions(newPartials, parentValue, true);

        partials.put(inputVertex, newPartials);

        return partials;
    }

    private static long[] checkInputIsSquareMatrix(long[] shape) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Can only invert a Matrix (received rank: " + shape.length + ")");
        }

        if (shape[0] != shape[1]) {
            throw new IllegalArgumentException("Can only invert a square Matrix (received: "
                + shape[0] + ", " + shape[1] + ")");
        }

        return shape;
    }

}
