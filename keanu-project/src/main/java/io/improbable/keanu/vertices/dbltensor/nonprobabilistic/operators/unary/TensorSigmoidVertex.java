package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

public class TensorSigmoidVertex extends TensorDoubleUnaryOpVertex {

    public TensorSigmoidVertex(DoubleTensorVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.unaryMinus().expInPlace().plusInPlace(1).reciprocalInPlace();
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        TensorDualNumber dualNumber = dualNumbers.get(inputVertex);
        DoubleTensor x = dualNumber.getValue();
        DoubleTensor xExp = x.exp();
        DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
        TensorPartialDerivatives infinitesimal = dualNumber.getPartialDerivatives().multiplyBy(dxdfx);
        return new TensorDualNumber(x.sigmoid(), infinitesimal);
    }

}
