package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;

public class AllFalseVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    @ExportVertexToPythonBindings
    public AllFalseVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor, ?> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l) {
        return l.allFalse();
    }
}
