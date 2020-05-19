package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

public class CastNumberToDoubleVertex<T extends NumberTensor> extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor>, VertexUnaryOp<Vertex<T, ?>> {

    private final Vertex<T, ?> inputVertex;
    private static final String INPUT_VERTEX_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastNumberToDoubleVertex(@LoadVertexParam(INPUT_VERTEX_NAME) Vertex<T, ?> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_VERTEX_NAME)
    public Vertex<T, ?> getInputVertex() {
        return inputVertex;
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().toDouble();
    }
}
