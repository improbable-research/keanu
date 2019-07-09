package io.improbable.keanu.vertices.model;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.BooleanModelResultVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleModelResultVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.IntegerModelResultVertex;

import java.util.Map;

public interface ModelVertex<T> extends NonProbabilistic<T> {

    void run();

    Map<VertexLabel, Vertex<? extends Tensor>> updateValues();

    boolean hasCalculated();

    <U, T extends Tensor<U, T>> T getModelOutputValue(VertexLabel label);

    default DoubleVertex getDoubleModelOutputVertex(VertexLabel label) {
        return new DoubleModelResultVertex(this, label);
    }

    default IntegerVertex getIntegerModelOutputVertex(VertexLabel label) {
        return new IntegerModelResultVertex(this, label);
    }

    default BooleanVertex getBooleanModelOutputVertex(VertexLabel label) {
        return new BooleanModelResultVertex(this, label);
    }

}
