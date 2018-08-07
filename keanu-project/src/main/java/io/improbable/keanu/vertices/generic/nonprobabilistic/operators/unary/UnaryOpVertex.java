package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;


import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class UnaryOpVertex<IN, OUT> extends Vertex<OUT> {

    protected final Vertex<IN> inputVertex;

    public UnaryOpVertex(Vertex<IN> inputVertex) {
        super(new NonProbabilisticValueUpdater<>(v -> ((UnaryOpVertex<IN, OUT>) v).op(inputVertex.getValue())), Observable.observableTypeFor(UnaryOpVertex.class));
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public OUT sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected abstract OUT op(IN a);
}

