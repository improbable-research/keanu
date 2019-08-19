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

import java.util.Collections;
import java.util.Map;

/**
 * Gradient calculations sourced from https://arxiv.org/pdf/1602.07527.pdf
 * @param <T>
 * @param <TENSOR>
 * @param <VERTEX>
 */
public class CholeskyDecompositionVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public CholeskyDecompositionVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.choleskyDecomposition();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();
        DoubleTensor L = getValue().toDouble();
        DoubleTensor LInverse = L.matrixInverse();

        DoubleTensor toPhi = LInverse.matrixMultiply(partial.matrixMultiply(LInverse.transpose()));

        DoubleTensor result = L.matrixMultiply(phi(toPhi));

        return Collections.singletonMap(inputVertex, new PartialDerivative(result));
    }

    private DoubleTensor phi(DoubleTensor input){
        final long[] shape = input.getShape();
        final long M = shape[shape.length-1];
        final long N = shape[shape.length-2];

        final DoubleTensor diag = DoubleTensor.create(0.5, new long[]{N}).diag();
        final DoubleTensor factor = DoubleTensor.ones(M, N).minus(diag).triLower(0);
        return input.times(factor);
    }
}