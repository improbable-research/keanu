package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.model.ModelResult;
import io.improbable.keanu.vertices.model.ModelResultProvider;
import io.improbable.keanu.vertices.model.ModelVertex;

/**
 * A non-probabilistic boolean vertex whose value is extracted from an upstream model vertex.
 */
public class BooleanModelResultVertex extends BooleanVertex implements ModelResultProvider<BooleanTensor>, NonProbabilistic<BooleanTensor>, NonSaveableVertex {

    private final ModelResult<BooleanTensor> delegate;

    public BooleanModelResultVertex(ModelVertex model, VertexLabel label) {
        super(Tensor.SCALAR_SHAPE);
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
    public BooleanTensor calculate() {
        return delegate.calculate();
    }
}
