package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTCondition;

public class DoubleCPTVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<CPTCondition, DoubleVertex> conditions;
    private final DoubleVertex defaultResult;

    public DoubleCPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                           Map<CPTCondition, DoubleVertex> conditions,
                           DoubleVertex defaultResult) {
        this.inputs = inputs;
        this.conditions = conditions;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.sample(random).scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.sample(random) : vertex.sample(random);
    }

    @Override
    public DoubleTensor calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? dualNumbers.get(defaultResult) : dualNumbers.get(vertex);
    }
}
