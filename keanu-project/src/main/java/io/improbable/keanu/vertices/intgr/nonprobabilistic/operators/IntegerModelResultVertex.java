package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ModelVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerModelResultVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    private ModelVertex model;
    private VertexLabel label;
    private boolean hasValue;

    public IntegerModelResultVertex(ModelVertex model, VertexLabel label) {
        this.model = model;
        this.label = label;
        this.hasValue = false;
        setParents((Vertex) model);
    }

    @Override
    public IntegerTensor getValue() {
        if (!hasValue) {
            model.calculate();
        }
        return model.getIntegerModelOutputValue(label);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return model.getIntegerModelOutputValue(label);
    }

    @Override
    public IntegerTensor calculate() {
        hasValue = true;
        return model.getIntegerModelOutputValue(label);
    }
}
