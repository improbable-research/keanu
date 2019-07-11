package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerLessThanOrEqualMaskVertex extends IntegerBinaryOpVertex {

    @ExportVertexToPythonBindings
    public IntegerLessThanOrEqualMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.lessThanOrEqualToMask(r);
    }
}
