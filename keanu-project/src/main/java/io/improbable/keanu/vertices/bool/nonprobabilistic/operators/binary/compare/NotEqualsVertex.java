package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    @ExportVertexToPythonBindings
    public NotEqualsVertex(@LoadVertexParam(A_NAME) Vertex<A> a, @LoadVertexParam(B_NAME) Vertex<B> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(A l, B r) {
        return l.elementwiseEquals(r).not();
    }

}
