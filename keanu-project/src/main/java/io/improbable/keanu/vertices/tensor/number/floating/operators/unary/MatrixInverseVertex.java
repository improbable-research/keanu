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
        DoubleTensor wrtOf = derivativeOfParentWithRespectToInputs.getWrtOf(inputVertex.getRank());
        DoubleTensor result = negatedValue.matrixMultiply(wrtOf).matrixMultiply(this.getValue().toDouble());

        return PartialDerivative.createFromWrtOf(result, this.getRank());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor parentValue = getValue().toDouble();
        DoubleTensor negativeValue = getValue().toDouble().unaryMinus();

        DoubleTensor p = negativeValue.transpose()
            .matrixMultiply(derivativeOfOutputWithRespectToSelf.get())
            .matrixMultiply(parentValue.transpose());

        partials.put(inputVertex, new PartialDerivative(p));
        return partials;
    }

    private static long[] checkInputIsSquareMatrix(long[] shape) {
        if (shape.length < 2) {
            throw new IllegalArgumentException("Can only invert a matrix or batch of matrices (received rank: " + shape.length + ")");
        }

        if (shape[shape.length - 1] != shape[shape.length - 2]) {
            throw new IllegalArgumentException("Can only invert a square matrix (received: "
                + shape[0] + ", " + shape[1] + ")");
        }

        return shape;
    }

}
