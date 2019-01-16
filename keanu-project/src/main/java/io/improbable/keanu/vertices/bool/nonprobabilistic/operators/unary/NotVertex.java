package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;

public class NotVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    @ExportVertexToPythonBindings
    public NotVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> a) {
        super(a.getShape(), a);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.not();
    }
}
