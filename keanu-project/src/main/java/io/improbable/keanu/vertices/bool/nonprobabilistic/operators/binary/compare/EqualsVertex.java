package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class EqualsVertex<TENSOR extends Tensor> extends BoolBinaryOpVertex<TENSOR, TENSOR> {

    @ExportVertexToPythonBindings
    public EqualsVertex(@LoadParentVertex(A_NAME) Vertex<TENSOR> a, @LoadParentVertex(B_NAME) Vertex<TENSOR> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(TENSOR l, TENSOR r) {
        return l.elementwiseEquals(r);
    }
}
