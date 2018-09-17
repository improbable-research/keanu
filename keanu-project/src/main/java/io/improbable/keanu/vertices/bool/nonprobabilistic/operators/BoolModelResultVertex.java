package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.model.ModelVertex;
import io.improbable.keanu.vertices.model.ModelResult;
import io.improbable.keanu.vertices.model.ModelResultProvider;

/**
 * A non-probabilistic boolean vertex whose value is extracted from an upstream model vertex.
 */
public class BoolModelResultVertex extends BoolVertex implements ModelResultProvider<BooleanTensor>, NonProbabilistic<BooleanTensor> {

    private final ModelResult<BooleanTensor> delegate;

    public BoolModelResultVertex(ModelVertex model, VertexLabel label) {
        delegate = new ModelResult<>(model, label);
        setParents((Vertex) model);
    }

    @Override
    public ModelVertex<BooleanTensor> getModel() {
        return delegate.getModel();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return delegate.sample(random);
    }

    @Override
    public void calculate() {
        setValue(delegate.calculate());
    }
}
