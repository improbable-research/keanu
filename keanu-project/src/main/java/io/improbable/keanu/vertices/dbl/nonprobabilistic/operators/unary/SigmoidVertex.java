package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class SigmoidVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the sigmoid function to a vertex.
     * The sigmoid function is a special case of the Logistic function.
     *
     * @param inputVertex the vertex
     */
    public SigmoidVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.unaryMinus().expInPlace().plusInPlace(1).reciprocalInPlace();
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        DoubleTensor x = inputVertex.getValue();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dxdfx, x.getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor sigmoidOfInput = getValue();
        //dSigmoid = sigmoid(x)*(1-sigmoid(x))
        DoubleTensor derivativeOfSigmoidWrtInput = sigmoidOfInput.minus(sigmoidOfInput.pow(2));

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(derivativeOfSigmoidWrtInput, this.getShape()));
        return partials;
    }

}
