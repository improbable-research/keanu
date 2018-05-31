package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;

public class TensorArcTan2Vertex extends TensorBinaryOpVertex {

    public TensorArcTan2Vertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return ;
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        return null;
    }
}
