package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class LessThanVertex<A extends NumberTensor, B extends NumberTensor> extends BoolBinaryOpVertex<A, B> {

    @ExportVertexToPythonBindings
    public LessThanVertex(@LoadParentVertex(A_NAME) Vertex<A> a, @LoadParentVertex(B_NAME) Vertex<B> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(A l, B r) {
        return l.lessThan(r);
    }

}
