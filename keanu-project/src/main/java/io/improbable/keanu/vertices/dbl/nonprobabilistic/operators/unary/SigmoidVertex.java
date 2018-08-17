package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
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
    protected DualNumber dualOp(DualNumber a) {
        DoubleTensor x = a.getValue();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
        PartialDerivatives infinitesimal = a.getPartialDerivatives().multiplyBy(dxdfx);
        return new DualNumber(x.sigmoid(), infinitesimal);
    }
}
