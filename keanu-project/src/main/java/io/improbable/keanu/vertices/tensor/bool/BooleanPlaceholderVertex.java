package io.improbable.keanu.vertices.tensor.bool;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;

public class BooleanPlaceholderVertex extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex,  LogProbGraph.PlaceholderVertex, NonProbabilistic<BooleanTensor>, Differentiable, NonSaveableVertex {

    private final BooleanVertex defaultVertex;

    public BooleanPlaceholderVertex(long... initialShape) {
        super(initialShape);
        defaultVertex = null;
    }

    public BooleanPlaceholderVertex(BooleanVertex defaultVertex) {
        super(defaultVertex.getShape());
        this.defaultVertex = defaultVertex;
    }

    @Override
    public BooleanTensor calculate() {
        if (hasValue()) {
            return getValue();
        } else if (defaultVertex != null) {
            return defaultVertex.getValue();
        } else {
            throw new IllegalStateException("Placeholders must be fed values");
        }
    }

}
