package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class NotBinaryVertex extends BooleanUnaryOpVertex {

    @ExportVertexToPythonBindings
    public NotBinaryVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex a) {
        super(a.getShape(), a);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.not();
    }
}
