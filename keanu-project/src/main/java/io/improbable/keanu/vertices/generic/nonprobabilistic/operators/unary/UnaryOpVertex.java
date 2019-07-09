package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.generic.GenericVertex;

public abstract class UnaryOpVertex<IN, OUT> extends GenericVertex<OUT> implements NonProbabilistic<OUT> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final IVertex<IN> inputVertex;

    public UnaryOpVertex(long[] shape, IVertex<IN> inputVertex) {
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
    public IVertex<IN> getInputVertex() {
        return inputVertex;
    }
}
