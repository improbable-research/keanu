package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

public class TensorArcTan2Vertex extends TensorDoubleBinaryOpVertex {

    public TensorArcTan2Vertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.atan2(b);
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        TensorDualNumber aDual = dualNumbers.get(a);
        TensorDualNumber bDual = dualNumbers.get(b);

        DoubleTensor denominator = ((b.getValue().pow(2)).timesInPlace((a.getValue().pow(2))));

        TensorPartialDerivatives thisInfA = aDual.getPartialDerivatives().multiplyBy(b.getValue().div(denominator));
        TensorPartialDerivatives thisInfB = bDual.getPartialDerivatives().multiplyBy((a.getValue().div(denominator)).unaryMinusInPlace());
        TensorPartialDerivatives newInf = thisInfA.add(thisInfB);
        return new TensorDualNumber(op(a.getValue(), b.getValue()), newInf);
    }
}
