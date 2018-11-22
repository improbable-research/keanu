package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.Vertex;

public class OrBinaryVertex extends BoolBinaryOpVertex<BooleanTensor, BooleanTensor> {

    @ExportVertexToPythonBindings
    public OrBinaryVertex(@LoadParentVertex(A_NAME) Vertex<BooleanTensor> a,
                          @LoadParentVertex(B_NAME) Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l, BooleanTensor r) {
        return l.or(r);
    }
}
