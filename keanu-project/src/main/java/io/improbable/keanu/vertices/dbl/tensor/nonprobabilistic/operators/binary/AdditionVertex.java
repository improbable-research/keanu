package io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.tensor.DoubleTensor;
import io.improbable.keanu.vertices.dbl.scaler.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.tensor.DoubleTensorVertex;

public class AdditionVertex extends BinaryOpVertex {

    public AdditionVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber getDualNumber(DualNumber aDual, DualNumber bDual) {
        return aDual.add(bDual);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a + b;
    }
}
