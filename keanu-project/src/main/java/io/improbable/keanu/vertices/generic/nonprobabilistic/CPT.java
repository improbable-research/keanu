package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
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
    public static CPTRawBuilder of(Vertex<? extends Tensor<Boolean>>... inputs) {
        return new CPTRawBuilder(Arrays.asList(inputs));
    }

    public static class CPTRawBuilder {

        private final List<Vertex<? extends Tensor<Boolean>>> inputs;

        public CPTRawBuilder(List<Vertex<? extends Tensor<Boolean>>> inputs) {
            this.inputs = inputs;
        }

        public CPTWhenRawBuilder when(Boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenRawBuilder(condition, inputs);
        }
    }

    public static class CPTWhenRawBuilder {

        private final CPTVertex.Condition condition;
        private final List<Vertex<? extends Tensor<Boolean>>> inputs;

        public CPTWhenRawBuilder(Boolean[] condition, List<Vertex<? extends Tensor<Boolean>>> inputs) {
            this.condition = new CPTVertex.Condition(condition);
            this.inputs = inputs;
        }

        public <T, OUT extends Tensor<T>> CPTBuilder<T, OUT> then(Vertex<OUT> thn) {
            Map<CPTVertex.Condition, Vertex<OUT>> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new CPTBuilder<>(inputs, conditions);
        }
    }

    public static class CPTBuilder<T, OUT extends Tensor<T>> {
        private final List<Vertex<? extends Tensor<Boolean>>> inputs;
        private final Map<CPTVertex.Condition, Vertex<OUT>> conditions;

        public CPTBuilder(List<Vertex<? extends Tensor<Boolean>>> inputs, Map<CPTVertex.Condition, Vertex<OUT>> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public CPTWhenBuilder<T, OUT> when(Boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenBuilder<>(new CPTVertex.Condition(condition), this);
        }

        public CPTVertex<OUT> orDefault(Vertex<OUT> defaultResult) {
            return new CPTVertex<>(inputs, conditions, defaultResult);
        }

        public static class CPTWhenBuilder<T, OUT extends Tensor<T>> {

            private final CPTVertex.Condition condition;
            private final CPTBuilder<T, OUT> builder;

            private CPTWhenBuilder(CPTVertex.Condition condition, CPTBuilder<T, OUT> builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public CPTBuilder<T, OUT> then(Vertex<OUT> thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }
        }
    }
}
