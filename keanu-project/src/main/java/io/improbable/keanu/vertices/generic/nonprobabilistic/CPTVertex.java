package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

import java.util.List;
import java.util.Map;

public class CPTVertex<OUT extends Tensor> extends Vertex<OUT> {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<CPTCondition, ? extends Vertex<OUT>> conditions;
    private final Vertex<OUT> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                     Map<CPTCondition, ? extends Vertex<OUT>> conditions,
                     Vertex<OUT> defaultResult) {
        super(new NonProbabilisticValueUpdater<>(v -> ((CPTVertex<OUT>) v).getDerivedValue()));
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public OUT sample(KeanuRandom random) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.sample(random).scalar());
        Vertex<OUT> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.sample(random) : vertex.sample(random);
    }

    public OUT getDerivedValue() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        Vertex<OUT> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

}
