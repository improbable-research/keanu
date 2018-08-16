package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class MultiplexerVertex<T> extends Vertex<T> {

    private final IntegerVertex selectorControlVertex;
    private final Vertex<T>[] selectVertices;

    public MultiplexerVertex(IntegerVertex selectorControlVertex, Vertex<T>... select) {
        super(new NonProbabilisticValueUpdater<>(v -> ((MultiplexerVertex<T>) v).getSelector().getValue()));

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

    private T getDerivedValue() {
        Vertex<T> selector = getSelector();
        return selector.getValue();
    }

    private Vertex<T> getSelector() {
        int optionGroupIdx = selectorControlVertex.getValue().scalar();
        return selectVertices[optionGroupIdx];
    }
}