package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CPTVertex<OUT extends Tensor> extends NonProbabilistic<OUT> {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<Condition, ? extends Vertex<OUT>> conditions;
    private final Vertex<OUT> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                     Map<Condition, ? extends Vertex<OUT>> conditions,
                     Vertex<OUT> defaultResult) {
        super(v -> ((CPTVertex<OUT>)v).getDerivedValue());
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public OUT sample(KeanuRandom random) {
        final Condition condition = getCondition((vertex) -> vertex.sample(random).scalar());
        Vertex<OUT> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.sample(random) : vertex.sample(random);
    }

    public OUT getDerivedValue() {
        final Condition condition = getCondition(v -> v.getValue().scalar());
        Vertex<OUT> vertex = conditions.get(condition);
        return vertex == null ? defaultResult.getValue() : vertex.getValue();
    }

    private Condition getCondition(Function<Vertex<? extends Tensor<Boolean>>, Boolean> mapper) {

        Boolean[] condition = new Boolean[inputs.size()];

        for (int i = 0; i < condition.length; i++) {
            condition[i] = mapper.apply(inputs.get(i));
        }

        return new Condition(condition);
    }

    public static class Condition {
        private final Boolean[] conditions;

        public Condition(Boolean[] condition) {
            this.conditions = condition;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Condition condition1 = (Condition) o;

            return Arrays.equals(conditions, condition1.conditions);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(conditions);
        }
    }

}
