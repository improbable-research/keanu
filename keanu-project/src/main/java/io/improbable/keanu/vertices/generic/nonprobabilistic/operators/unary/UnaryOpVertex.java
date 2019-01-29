package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;


import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericVertex;

public abstract class UnaryOpVertex<IN, OUT> extends GenericVertex<OUT> implements NonProbabilistic<OUT> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final Vertex<IN> inputVertex;

    public UnaryOpVertex(long[] shape, Vertex<IN> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    @Override
    public OUT calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract OUT op(IN a);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<IN> getInputVertex() {
        return inputVertex;
    }
}
