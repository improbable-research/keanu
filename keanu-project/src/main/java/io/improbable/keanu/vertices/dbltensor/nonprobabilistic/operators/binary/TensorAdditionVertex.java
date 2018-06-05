package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;

public class TensorAdditionVertex extends TensorDoubleBinaryOpVertex {

    public TensorAdditionVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    public TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        TensorDualNumber aDual = dualNumbers.get(a);
        TensorDualNumber bDual = dualNumbers.get(b);
        return aDual.plus(bDual);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.plus(b);
    }
}
