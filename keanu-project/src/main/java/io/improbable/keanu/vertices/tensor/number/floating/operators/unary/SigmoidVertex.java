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

public class SigmoidVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * Applies the sigmoid function to a vertex.
     * The sigmoid function is a special case of the Logistic function.
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public SigmoidVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.sigmoid();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        DoubleTensor x = inputVertex.getValue().toDouble();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2.0));
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dxdfx);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        DoubleTensor sigmoidOfInput = getValue().toDouble();
        //dSigmoid = sigmoid(x)*(1-sigmoid(x))
        DoubleTensor derivativeOfSigmoidWrtInput = sigmoidOfInput.minus(sigmoidOfInput.pow(2));

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(derivativeOfSigmoidWrtInput));
        return partials;
    }

}
