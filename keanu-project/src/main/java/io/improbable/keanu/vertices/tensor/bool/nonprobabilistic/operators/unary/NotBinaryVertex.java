package io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;

public class NotBinaryVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    @ExportVertexToPythonBindings
    public NotBinaryVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor, ?> a) {
        super(a.getShape(), a);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.not();
    }
}
