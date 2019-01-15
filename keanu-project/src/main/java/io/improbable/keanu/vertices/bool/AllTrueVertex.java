package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolUnaryOpVertex;

public class AllTrueVertex extends BoolUnaryOpVertex<BooleanTensor> {

    @ExportVertexToPythonBindings
    public AllTrueVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> boolVertex) {
        super(boolVertex);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return BooleanTensor.create(value.allTrue(), value.getShape());
    }
}
