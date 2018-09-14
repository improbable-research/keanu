package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;

/**
 * A non-probabilistic boolean vertex whose value is extracted from an upstream model vertex.
 */
public class BoolModelResultVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    private ModelVertex model;
    private VertexLabel label;

    public BoolModelResultVertex(ModelVertex model, VertexLabel label) {
        this.model = model;
        this.label = label;
        setParents((Vertex) model);
    }

    @Override
    public BooleanTensor getValue() {
        if (!model.hasCalculated()) {
            model.calculate();
        }
        return model.getBooleanModelOutputValue(label);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return model.getBooleanModelOutputValue(label);
    }

    @Override
    public BooleanTensor calculate() {
        return sample();
    }
}
