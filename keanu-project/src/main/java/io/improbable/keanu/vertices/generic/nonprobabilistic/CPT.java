package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CPT {

    private CPT() {
    }

    private static final String WHEN_CONDITION_SIZE_MISMATCH = "The 'when' condition size does not match input count";

    @SafeVarargs
    public static CPTRawBuilder of(Vertex<Boolean>... inputs) {
        return new CPTRawBuilder(Arrays.asList(inputs));
    }

    public static class CPTRawBuilder {

        private final List<Vertex<Boolean>> inputs;

        public CPTRawBuilder(List<Vertex<Boolean>> inputs) {
            this.inputs = inputs;
        }

        public CPTWhenRawBuilder when(boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenRawBuilder(condition, inputs);
        }
    }

    public static class CPTWhenRawBuilder {

        private final CPTVertex.Condition condition;
        private final List<Vertex<Boolean>> inputs;

        public CPTWhenRawBuilder(boolean[] condition, List<Vertex<Boolean>> inputs) {
            this.condition = new CPTVertex.Condition(condition);
            this.inputs = inputs;
        }

        public <T> CPTBuilder<T> then(Vertex<T> thn) {
            Map<CPTVertex.Condition, Vertex<T>> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new CPTBuilder<>(inputs, conditions);
        }
    }

    public static class CPTBuilder<T> {
        private final List<Vertex<Boolean>> inputs;
        private final Map<CPTVertex.Condition, Vertex<T>> conditions;

        public CPTBuilder(List<Vertex<Boolean>> inputs, Map<CPTVertex.Condition, Vertex<T>> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public CPTWhenBuilder<T> when(boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenBuilder<>(new CPTVertex.Condition(condition), this);
        }

        public CPTVertex<T> orDefault(Vertex<T> defaultResult) {
            return new CPTVertex<>(inputs, conditions, defaultResult);
        }

        public static class CPTWhenBuilder<T> {

            private final CPTVertex.Condition condition;
            private final CPTBuilder<T> builder;

            private CPTWhenBuilder(CPTVertex.Condition condition, CPTBuilder<T> builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public CPTBuilder<T> then(Vertex<T> thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }
        }
    }
}
