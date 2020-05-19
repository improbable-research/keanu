package io.improbable.keanu.vertices.tensor.number.floating.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;

public class DoublePlaceholderVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, LogProbGraph.PlaceholderVertex, NonProbabilistic<DoubleTensor>, Differentiable, NonSaveableVertex {

    private final DoubleVertex defaultVertex;

    public DoublePlaceholderVertex(long... initialShape) {
        super(initialShape);
        defaultVertex = null;
    }

    public DoublePlaceholderVertex(DoubleVertex defaultVertex) {
        super(defaultVertex.getShape());
        this.defaultVertex = defaultVertex;
    }

    @Override
    public DoubleTensor calculate() {
        if (hasValue()) {
            return getValue();
        } else if (defaultVertex != null) {
            return defaultVertex.getValue();
        } else {
            throw new IllegalStateException("Placeholders must be fed values");
        }
    }

}
