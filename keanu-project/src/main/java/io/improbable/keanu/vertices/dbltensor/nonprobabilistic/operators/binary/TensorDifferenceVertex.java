package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;

public class TensorDifferenceVertex extends TensorBinaryOpVertex {

    public TensorDifferenceVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    public TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        TensorDualNumber aDual = dualNumbers.get(a);
        TensorDualNumber bDual = dualNumbers.get(b);
        return aDual.subtract(bDual);
    }

    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.minus(b);
    }
}
