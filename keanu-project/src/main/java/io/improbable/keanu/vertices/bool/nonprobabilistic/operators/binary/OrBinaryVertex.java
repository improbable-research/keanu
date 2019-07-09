package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;

@DisplayInformationForOutput(displayName = "OR")
public class OrBinaryVertex extends BooleanBinaryOpVertex<BooleanTensor, BooleanTensor> {

    @ExportVertexToPythonBindings
    public OrBinaryVertex(@LoadVertexParam(A_NAME) Vertex<BooleanTensor> a,
                          @LoadVertexParam(B_NAME) Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l, BooleanTensor r) {
        return l.or(r);
    }
}
