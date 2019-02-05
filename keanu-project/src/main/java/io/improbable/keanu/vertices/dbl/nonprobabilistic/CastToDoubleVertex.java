package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class CastToDoubleVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends NumberTensor> inputVertex;
    private static final String INPUT_VERTEX_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastToDoubleVertex(@LoadVertexParam(INPUT_VERTEX_NAME) Vertex<? extends NumberTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_VERTEX_NAME)
    public Vertex<? extends NumberTensor> getInputVertex() {
        return inputVertex;
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().toDouble();
    }
}
