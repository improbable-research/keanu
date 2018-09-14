package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;
import io.improbable.keanu.vertices.Vertex;

/**
 * A non-probabilistic double vertex whose value is extracted from an upstream model vertex.
 */
public class DoubleModelResultVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private ModelVertex model;
    private VertexLabel label;

    public DoubleModelResultVertex(ModelVertex model, VertexLabel label) {
        this.model = model;
        this.label = label;
        setParents((Vertex) model);
    }

    @Override
    public DoubleTensor getValue() {
        if (!model.hasCalculated()) {
            model.calculate();
        }
        return model.getDoubleModelOutputValue(label);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return model.getDoubleModelOutputValue(label);
    }

    @Override
    public DoubleTensor calculate() {
        return sample();
    }

}
