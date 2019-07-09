package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

import java.util.List;
import java.util.Map;

public class CPTVertex<T> extends GenericTensorVertex<T> implements NonProbabilistic<GenericTensor<T>>, NonSaveableVertex {

    private final List<IVertex<? extends Tensor<?, ?>>> inputs;
    private final Map<CPTCondition, ? extends IVertex<GenericTensor<T>>> conditions;
    private final IVertex<GenericTensor<T>> defaultResult;

    public CPTVertex(List<IVertex<? extends Tensor<?, ?>>> inputs,
                     Map<CPTCondition, ? extends IVertex<GenericTensor<T>>> conditions,
                     IVertex<GenericTensor<T>> defaultResult) {
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
        IVertex<GenericTensor<T>> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
