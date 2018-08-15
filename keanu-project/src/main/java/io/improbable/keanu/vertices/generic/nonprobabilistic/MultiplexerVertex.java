package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class MultiplexerVertex<T> extends Vertex<T> implements NonProbabilistic<T> {

    private final IntegerVertex selectorControlVertex;
    private final Vertex<T>[] selectVertices;

    public MultiplexerVertex(IntegerVertex selectorControlVertex, Vertex<T>... select) {

        if (!TensorShape.isScalar(selectorControlVertex.getShape())) {
            throw new IllegalArgumentException("Select control must be scalar integer");
        }

        this.selectVertices = select;
        this.selectorControlVertex = selectorControlVertex;
        setParents(select);
        addParent(selectorControlVertex);
    }

    private Vertex<T> getSelector() {
        int optionGroupIdx = selectorControlVertex.getValue().scalar();
        return selectVertices[optionGroupIdx];
    }

    @Override
    public T sample(KeanuRandom random) {
        Vertex<T> selector = getSelector();
        return selector.sample(random);
    }

    @Override
    public T calculate() {
        Vertex<T> selector = getSelector();
        return selector.getValue();
    }
}