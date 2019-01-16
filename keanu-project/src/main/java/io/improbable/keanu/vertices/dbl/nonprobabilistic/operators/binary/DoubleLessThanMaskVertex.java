package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

public class DoubleLessThanMaskVertex extends DoubleBinaryOpVertex {
    public DoubleLessThanMaskVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.getLessThanMask(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInput, PartialDerivative dRightWrtInput) {
        return null;
    }
}

