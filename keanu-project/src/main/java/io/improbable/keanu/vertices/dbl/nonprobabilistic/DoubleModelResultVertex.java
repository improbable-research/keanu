package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.model.ModelResult;
import io.improbable.keanu.vertices.model.ModelResultProvider;
import io.improbable.keanu.vertices.model.ModelVertex;

/**
 * A non-probabilistic double vertex whose value is extracted from an upstream model vertex.
 */
public class DoubleModelResultVertex extends DoubleVertex implements ModelResultProvider<DoubleTensor>, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final ModelResult<DoubleTensor> delegate;

    public DoubleModelResultVertex(ModelVertex model, VertexLabel label) {
        super(Tensor.SCALAR_SHAPE);
        this.delegate = new ModelResult<>(model, label);
        setParents((Vertex) model);
    }

    @Override
    public ModelVertex<DoubleTensor> getModel() {
        return delegate.getModel();
    }

    @Override
    public DoubleTensor calculate() {
        return delegate.calculate();
    }
}
