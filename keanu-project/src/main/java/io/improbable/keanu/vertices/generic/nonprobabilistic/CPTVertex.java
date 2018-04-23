package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CPTVertex<T> extends NonProbabilistic<T> {

    private final List<Vertex<Boolean>> inputs;
    private final Map<Condition, Vertex<T>> conditions;
    private final Vertex<T> defaultResult;

    public CPTVertex(List<Vertex<Boolean>> inputs, Map<Condition, Vertex<T>> conditions, Vertex<T> defaultResult) {
        this.conditions = conditions;
        this.inputs = inputs;
        this.defaultResult = defaultResult;
        addParents(inputs);
        addParents(conditions.values());
        addParent(defaultResult);
    }

    @Override
    public T sample() {
        final Condition condition = getCondition(Vertex::sample);
        return conditions.getOrDefault(condition, defaultResult).sample();
    }

    @Override
    public T getDerivedValue() {
        final Condition condition = getCondition(Vertex::getValue);
        return conditions.getOrDefault(condition, defaultResult).getValue();
    }

    private Condition getCondition(Function<Vertex<Boolean>, Boolean> mapper) {

        boolean[] condition = new boolean[inputs.size()];

        for (int i = 0; i < condition.length; i++) {
            condition[i] = mapper.apply(inputs.get(i));
        }

        return new Condition(condition);
    }

    public static class Condition {
        private final boolean[] condition;

        public Condition(boolean[] condition) {
            this.condition = condition;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Condition condition1 = (Condition) o;

            return Arrays.equals(condition, condition1.condition);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(condition);
        }
    }

}
