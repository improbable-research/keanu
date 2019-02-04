package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

import java.util.List;
import java.util.Map;

public class CPTVertex<OUT extends Tensor> extends GenericTensorVertex<OUT> implements NonProbabilistic<OUT>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<CPTCondition, ? extends Vertex<OUT>> conditions;
    private final Vertex<OUT> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                     Map<CPTCondition, ? extends Vertex<OUT>> conditions,
                     Vertex<OUT> defaultResult) {
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public OUT calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        Vertex<OUT> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
