package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

public class TensorSigmoidVertex extends TensorDoubleUnaryOpVertex {

    public TensorSigmoidVertex(DoubleTensorVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.unaryMinus().exp().plus(1).reciprocal();
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        TensorDualNumber dualNumber = dualNumbers.get(inputVertex);
        DoubleTensor x = dualNumber.getValue();
        DoubleTensor dxdfx = x.exp().div(x.exp().plus(1).pow(2));
        TensorPartialDerivatives infinitesimal = dualNumber.getPartialDerivatives().multiplyBy(dxdfx);
        return new TensorDualNumber(x.sigmoid(), infinitesimal);
    }

}
