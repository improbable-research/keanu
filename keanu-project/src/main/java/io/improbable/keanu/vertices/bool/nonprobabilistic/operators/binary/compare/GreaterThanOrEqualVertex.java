package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;

public class GreaterThanOrEqualVertex<A extends NumberTensor, B extends NumberTensor> extends BooleanBinaryOpVertex<A, B> {

    @ExportVertexToPythonBindings
    public GreaterThanOrEqualVertex(@LoadVertexParam(A_NAME) Vertex<A> a, @LoadVertexParam(B_NAME) Vertex<B> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(A l, B r) {
        return l.greaterThanOrEqual(r);
    }
}
