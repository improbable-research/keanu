package io.improbable.keanu.vertices.tensor.number.fixed.intgr;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.PlaceholderVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;

public class IntegerPlaceholderVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, PlaceholderVertex, NonProbabilistic<IntegerTensor>, Differentiable, NonSaveableVertex {

    private final IntegerVertex defaultVertex;

    public IntegerPlaceholderVertex(long... initialShape) {
        super(initialShape);
        defaultVertex = null;
    }

    public IntegerPlaceholderVertex(IntegerVertex defaultVertex) {
        super(defaultVertex.getShape());
        this.defaultVertex = defaultVertex;
    }

    @Override
    public IntegerTensor calculate() {
        if (hasValue()) {
            return getValue();
        } else if (defaultVertex != null) {
            return defaultVertex.getValue();
        } else {
            throw new IllegalStateException("Placeholders must be fed values");
        }
    }

}
