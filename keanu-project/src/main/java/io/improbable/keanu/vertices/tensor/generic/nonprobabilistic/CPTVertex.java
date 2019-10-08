package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericTensorVertex;

import java.util.List;
import java.util.Map;

public class CPTVertex<T> extends VertexImpl<GenericTensor<T>, GenericTensorVertex<T>> implements GenericTensorVertex<T>, NonProbabilistic<GenericTensor<T>>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<?, ?>, ?>> inputs;
    private final Map<CPTCondition, ? extends Vertex<GenericTensor<T>, ?>> conditions;
    private final Vertex<GenericTensor<T>, ?> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<?, ?>, ?>> inputs,
                     Map<CPTCondition, ? extends Vertex<GenericTensor<T>, ?>> conditions,
                     Vertex<GenericTensor<T>, ?> defaultResult) {
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public GenericTensor<T> calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        Vertex<GenericTensor<T>, ?> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
