package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class CastNumberToBooleanVertex<T extends NumberTensor> extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexUnaryOp<Vertex<T, ?>> {

    private final Vertex<T, ?> inputVertex;
    private static final String INPUT_VERTEX_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastNumberToBooleanVertex(@LoadVertexParam(INPUT_VERTEX_NAME) Vertex<T, ?> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_VERTEX_NAME)
    public Vertex<T, ?> getInputVertex() {
        return inputVertex;
    }

    @Override
    public BooleanTensor calculate() {
        return inputVertex.getValue().toBoolean();
    }
}
