package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class MatrixInverseVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    @ExportVertexToPythonBindings
    public MatrixInverseVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(checkInputIsSquareMatrix(inputVertex.getShape()), inputVertex, inputVertex.ofType());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.matrixInverse();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        //dc = -A^-1 * da * A^-1
        DoubleTensor negatedValue = this.getValue().toDouble().unaryMinus();
        PartialDerivative partial = PartialDerivative.matrixMultiplyAlongOfDimensions(derivativeOfParentWithRespectToInputs, negatedValue, false);
        partial = PartialDerivative.matrixMultiplyAlongOfDimensions(partial, this.getValue().toDouble(), true);
        return partial;
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor parentValue = getValue().toDouble();
        DoubleTensor negativeValue = getValue().toDouble().unaryMinus();

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
