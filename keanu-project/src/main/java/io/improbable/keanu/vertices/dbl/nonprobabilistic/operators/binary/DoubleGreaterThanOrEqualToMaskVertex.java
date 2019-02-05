package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class DoubleGreaterThanOrEqualToMaskVertex extends DoubleBinaryOpVertex {

    @ExportVertexToPythonBindings
    public DoubleGreaterThanOrEqualToMaskVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                                                @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.getGreaterThanOrEqualToMask(r);
    }

}
