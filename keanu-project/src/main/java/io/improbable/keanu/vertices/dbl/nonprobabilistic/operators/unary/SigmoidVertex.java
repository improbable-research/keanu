package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class SigmoidVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the sigmoid function to a vertex.
     * The sigmoid function is a special case of the Logistic function.
     *
     * @param inputVertex the vertex
     */
    public SigmoidVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.unaryMinus().expInPlace().plusInPlace(1).reciprocalInPlace();
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber dualNumber = dualNumbers.get(inputVertex);
        DoubleTensor x = dualNumber.getValue();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
        PartialDerivatives infinitesimal = dualNumber.getPartialDerivatives().multiplyBy(dxdfx);
        return new DualNumber(x.sigmoid(), infinitesimal);
    }

    @Override
    protected Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor sigmoidOfInput = getValue();
        //dSigmoid = sigmoid(x)*(1-sigmoid(x))
        DoubleTensor derivativeOfSigmoidWrtInput = sigmoidOfInput.minus(sigmoidOfInput.pow(2));

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyBy(derivativeOfSigmoidWrtInput));
        return partials;
    }

}
