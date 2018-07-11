package io.improbable.keanu.vertices.generic.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class Probabilistic<T> extends Vertex<T> {

    public Probabilistic() {
        super(new ProbabilisticValueUpdater<>());
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
