package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTCondition;

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
    public DoubleTensor calculate() {
        final CPTCondition condition = CPTCondition.from(inputs, v -> v.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex vertex = conditions.get(condition);
        return vertex == null ? derivativeOfParentsWithRespectToInput.get(defaultResult) : derivativeOfParentsWithRespectToInput.get(vertex);
    }

    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        final CPTCondition condition = CPTCondition.from(inputs, (vertex) -> vertex.getValue().scalar());
        DoubleVertex conditionVertex = conditions.get(condition);

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        for (Vertex v : conditions.values()) {
            if (v == conditionVertex) {
                partials.put(v, derivativeOfOutputWithRespectToSelf);
            }
        }

        if (conditionVertex == null) {
            partials.put(defaultResult, derivativeOfOutputWithRespectToSelf);
        }

        return partials;
    }
}
