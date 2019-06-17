package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

import java.util.List;
import java.util.Map;

public class CPTVertex<T, TENSOR extends Tensor<T, TENSOR>> extends GenericTensorVertex<T, TENSOR> implements NonProbabilistic<TENSOR>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<?, ?>>> inputs;
    private final Map<CPTCondition, ? extends Vertex<TENSOR>> conditions;
    private final Vertex<TENSOR> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<?, ?>>> inputs,
                     Map<CPTCondition, ? extends Vertex<TENSOR>> conditions,
                     Vertex<TENSOR> defaultResult) {
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public TENSOR calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        Vertex<TENSOR> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
