package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericVertex;

public abstract class UnaryOpVertex<IN, OUT> extends VertexImpl<OUT, GenericVertex<OUT>> implements GenericVertex<OUT>, NonProbabilistic<OUT> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final Vertex<IN, ?> inputVertex;

    public UnaryOpVertex(long[] shape, Vertex<IN, ?> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract OUT op(IN a);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<IN, ?> getInputVertex() {
        return inputVertex;
    }
}
