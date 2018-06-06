package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CPTVertex<OUT extends Tensor> extends NonProbabilistic<OUT> {

    private final List<Vertex<? extends Tensor<Boolean>>> inputs;
    private final Map<Condition, Vertex<OUT>> conditions;
    private final Vertex<OUT> defaultResult;

    public CPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                     Map<Condition, Vertex<OUT>> conditions,
                     Vertex<OUT> defaultResult) {
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
        return conditions.getOrDefault(condition, defaultResult).sample(random);
    }

    @Override
    public OUT getDerivedValue() {
        final Condition condition = getCondition(v -> v.getValue().scalar());
        return conditions.getOrDefault(condition, defaultResult).getValue();
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
