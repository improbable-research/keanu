package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;

public class TensorLogVertex extends TensorDoubleUnaryOpVertex {

    public TensorLogVertex(DoubleTensorVertex inputVertex) {
        super(inputVertex);
    }

    protected DoubleTensor op(DoubleTensor a) {
        return a.log();
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).log();
    }
}
