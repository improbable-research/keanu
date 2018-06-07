package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class MultiplexerVertex<T> extends NonProbabilistic<T> {

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

    @Override
    public T sample(KeanuRandom random) {
        return getDerivedValue();
    }

    @Override
    public T getDerivedValue() {
        Vertex<T> selector = getSelector();
        return selector.getValue();
    }

    private Vertex<T> getSelector() {
        int optionGroupIdx = selectorControlVertex.getValue().scalar();
        return selectVertices[optionGroupIdx];
    }
}