package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Random;

public class MultiplexerVertex<T> extends NonProbabilistic<T> {

    private final IntegerVertex selectorControlVertex;
    private final Vertex<T>[] selectVertices;

    public MultiplexerVertex(IntegerVertex selectorControlVertex, Vertex<T>... select) {
        this.selectVertices = select;
        this.selectorControlVertex = selectorControlVertex;
        setParents(select);
        addParent(selectorControlVertex);
    }

    @Override
    public T sample(Random random) {
        return getDerivedValue();
    }

    @Override
    public T lazyEval() {
        setValue(getDerivedValue());
        return getValue();
    }

    @Override
    public T getDerivedValue() {
        Vertex<T> selector = getSelector();
        return selector.getValue();
    }

    private Vertex<T> getSelector() {
        int optionGroupIdx = selectorControlVertex.getValue();
        return selectVertices[optionGroupIdx];
    }
}