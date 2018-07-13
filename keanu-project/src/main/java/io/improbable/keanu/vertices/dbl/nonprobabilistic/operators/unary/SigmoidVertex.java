package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class SigmoidVertex extends DoubleUnaryOpVertex implements Differentiable {

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
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        DualNumber dualNumber = dualNumbers.get(inputVertex);
        DoubleTensor x = dualNumber.getValue();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
        PartialDerivatives infinitesimal = dualNumber.getPartialDerivatives().multiplyBy(dxdfx);
        return new DualNumber(x.sigmoid(), infinitesimal);
    }

}
