package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTCondition;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DoubleCPTVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<CPTCondition, DoubleVertex> conditions;
    private final DoubleVertex defaultResult;

    public DoubleCPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                           Map<CPTCondition, DoubleVertex> conditions,
                           DoubleVertex defaultResult) {
        super(defaultResult.getShape());
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
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? derivativeOfParentsWithRespectToInputs.get(defaultResult) : derivativeOfParentsWithRespectToInputs.get(vertex);
    }

    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex conditionVertex = conditions.get(condition);

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        for (Vertex v : conditions.values()) {
            if (v == conditionVertex) {
                partials.put(v, derivativeOfOutputsWithRespectToSelf);
            } else {
                partials.put(v, PartialDerivatives.OF_CONSTANT);
            }
        }

        if (conditionVertex == null) {
            partials.put(defaultResult, derivativeOfOutputsWithRespectToSelf);
        } else {
            partials.put(defaultResult, PartialDerivatives.OF_CONSTANT);
        }

        return partials;
    }
}
