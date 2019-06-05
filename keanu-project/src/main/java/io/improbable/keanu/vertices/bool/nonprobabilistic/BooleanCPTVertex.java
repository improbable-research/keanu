package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTCondition;

import java.util.List;
import java.util.Map;

public class BooleanCPTVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<?>>> inputs;
    private final Map<CPTCondition, BooleanVertex> conditions;
    private final BooleanVertex defaultResult;

    public BooleanCPTVertex(List<Vertex<? extends Tensor<?>>> inputs,
                            Map<CPTCondition, BooleanVertex> conditions,
                            BooleanVertex defaultResult) {
        super(defaultResult.getShape());
        this.inputs = inputs;
        this.conditions = conditions;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public BooleanTensor calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        BooleanVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }
}
