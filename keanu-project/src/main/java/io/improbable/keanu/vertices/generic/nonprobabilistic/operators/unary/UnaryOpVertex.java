package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;


import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

import java.util.Random;

public abstract class UnaryOpVertex<IN, OUT> extends NonProbabilistic<OUT> {

    protected final Vertex<IN> inputVertex;

    public UnaryOpVertex(Vertex<IN> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT sample(Random random) {
        return op(inputVertex.sample(random));
    }

    public OUT getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract OUT op(IN a);
}

