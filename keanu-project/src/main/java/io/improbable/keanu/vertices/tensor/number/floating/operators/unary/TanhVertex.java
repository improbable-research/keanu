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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class TanhVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public TanhVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.tanh();
    }

    private DoubleTensor dTanh(final DoubleTensor tanh) {
        //dTanhdx = 1 - tanh^2
        return tanh.times(tanh).reverseMinusInPlace(1.0);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        final DoubleTensor tanh = getValue().toDouble();
        return derivativeOfParentWithRespectToInputs.multiply(dTanh(tanh));
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        final DoubleTensor tanh = getValue().toDouble();
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.multiply(dTanh(tanh)));
        return partials;
    }
}