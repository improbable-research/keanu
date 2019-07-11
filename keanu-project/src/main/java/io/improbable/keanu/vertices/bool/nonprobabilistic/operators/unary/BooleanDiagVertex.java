package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class BooleanDiagVertex extends BooleanUnaryOpVertex {

    @ExportVertexToPythonBindings
    public BooleanDiagVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l) {
        return l.diag();
    }
}
