package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;

public class BooleanPlaceholderVertex extends BooleanVertex implements LogProbGraph.PlaceholderVertex, NonProbabilistic<BooleanTensor>, Differentiable, NonSaveableVertex {

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
