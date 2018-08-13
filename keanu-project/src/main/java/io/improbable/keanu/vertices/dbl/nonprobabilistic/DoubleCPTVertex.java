package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTCondition;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

import java.util.List;
import java.util.Map;

public class DoubleCPTVertex extends DoubleVertex implements Differentiable {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<CPTCondition, DoubleVertex> conditions;
    private final DoubleVertex defaultResult;

    public DoubleCPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                           Map<CPTCondition, DoubleVertex> conditions,
                           DoubleVertex defaultResult) {
        super(new NonProbabilisticValueUpdater<>(v -> ((DoubleCPTVertex) v).op()));
        this.inputs = inputs;
        this.conditions = conditions;
        this.defaultResult = defaultResult;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.sample(random).scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.sample(random) : vertex.sample(random);
    }

    protected DoubleTensor op() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        throw new UnsupportedOperationException("if is non-differentiable");
    }

}
