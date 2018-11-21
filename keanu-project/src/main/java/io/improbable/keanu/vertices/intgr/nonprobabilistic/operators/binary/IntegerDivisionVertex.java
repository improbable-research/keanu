package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;


public class IntegerDivisionVertex extends IntegerBinaryOpVertex {

    /**
     * Divides one vertex by another
     *
     * @param left a vertex to be divided
     * @param right a vertex to divide by
     */
    @ExportVertexToPythonBindings
    public IntegerDivisionVertex(@LoadParentVertex(LEFT_NAME) IntegerVertex left, @LoadParentVertex(RIGHT_NAME) IntegerVertex right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.div(r);
    }
}
