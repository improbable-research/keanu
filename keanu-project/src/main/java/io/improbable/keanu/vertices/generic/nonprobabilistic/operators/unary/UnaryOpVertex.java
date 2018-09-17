package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;


import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class UnaryOpVertex<IN, OUT> extends Vertex<OUT> implements NonProbabilistic<OUT> {

    protected final Vertex<IN> inputVertex;

    public UnaryOpVertex(Vertex<IN> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    @Override
    public void calculate() {
        setValue(op(inputVertex.getValue()));
    }

    protected abstract OUT op(IN a);
}

