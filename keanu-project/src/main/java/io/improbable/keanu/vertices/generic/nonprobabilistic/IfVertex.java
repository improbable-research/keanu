package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

public class IfVertex<T> extends NonProbabilistic<T> {

    private final Vertex<Boolean> predicate;
    private final Vertex<T> thn;
    private final Vertex<T> els;

    public IfVertex(Vertex<Boolean> predicate, Vertex<T> thn, Vertex<T> els) {
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @Override
    public T sample() {
        return op(predicate.sample(), thn.sample(), els.sample());
    }

    @Override
    public T getDerivedValue() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private T op(boolean predicate, T thn, T els) {
        return predicate ? thn : els;
    }

}
