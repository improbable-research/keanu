package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.model.ModelVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.model.ModelResult;
import io.improbable.keanu.vertices.model.ModelResultProvider;

/**
 * A non-probabilistic integer vertex whose value is extracted from an upstream model vertex.
 */
public class IntegerModelResultVertex extends IntegerVertex implements ModelResultProvider<IntegerTensor>, NonProbabilistic<IntegerTensor> {

    private final ModelResult<IntegerTensor> delegate;

    public IntegerModelResultVertex(ModelVertex model, VertexLabel label) {
        this.delegate = new ModelResult<>(model, label);
        setParents((Vertex) model);
    }

    @Override
    public ModelVertex<IntegerTensor> getModel() {
        return delegate.getModel();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return delegate.sample(random);
    }

    @Override
    public void calculate() {
        setValue(delegate.calculate());
    }
}
