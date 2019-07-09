package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BooleanBinaryOpVertex<A, B> {

    @ExportVertexToPythonBindings
    public NotEqualsVertex(@LoadVertexParam(A_NAME) IVertex<A> a, @LoadVertexParam(B_NAME) IVertex<B> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(A l, B r) {
        return l.elementwiseEquals(r).not();
    }
}
