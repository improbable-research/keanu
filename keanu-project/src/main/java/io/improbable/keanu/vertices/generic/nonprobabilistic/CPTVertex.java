package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

import java.util.List;
import java.util.Map;

public class CPTVertex<T> extends GenericTensorVertex<T> implements NonProbabilistic<Tensor<T>>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<?>>> inputs;
    private final Map<CPTCondition, ? extends Vertex<Tensor<T>>> conditions;
    private final Vertex<Tensor<T>> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<?>>> inputs,
                     Map<CPTCondition, ? extends Vertex<Tensor<T>>> conditions,
                     Vertex<Tensor<T>> defaultResult) {
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public Tensor<T> calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        Vertex<Tensor<T>> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
