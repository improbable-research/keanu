package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Random;

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
    public T sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public T getDerivedValue() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private T op(boolean predicate, T thn, T els) {
        return predicate ? thn : els;
    }

}
