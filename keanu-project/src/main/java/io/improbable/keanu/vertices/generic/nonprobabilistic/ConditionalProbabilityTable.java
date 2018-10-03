package io.improbable.keanu.vertices.generic.nonprobabilistic;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleCPTVertex;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ConditionalProbabilityTable {

    private ConditionalProbabilityTable() {}

    private static final String WHEN_CONDITION_SIZE_MISMATCH =
            "The 'when' condition size does not match input count";

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
            return new CPTWhenRawBuilder(ImmutableList.copyOf(condition), inputs);
        }
    }

    public static class CPTWhenRawBuilder {

        private final CPTCondition condition;
        private final List<Vertex<? extends Tensor<Boolean>>> inputs;

        public CPTWhenRawBuilder(
                List<Boolean> condition, List<Vertex<? extends Tensor<Boolean>>> inputs) {
            this.condition = new CPTCondition(condition);
            this.inputs = inputs;
        }

        public <T, OUT extends Tensor<T>> CPTBuilder<T, OUT> then(Vertex<OUT> thn) {
            Map<CPTCondition, Vertex<OUT>> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new CPTBuilder<>(inputs, conditions);
        }

        public DoubleCPTBuilder then(DoubleVertex thn) {
            Map<CPTCondition, DoubleVertex> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new DoubleCPTBuilder(inputs, conditions);
        }

        public DoubleCPTBuilder then(double thn) {
            return then(ConstantVertex.of(thn));
        }
    }

    public static class DoubleCPTBuilder {
        private final List<Vertex<? extends Tensor<Boolean>>> inputs;
        private final Map<CPTCondition, DoubleVertex> conditions;

        public DoubleCPTBuilder(
                List<Vertex<? extends Tensor<Boolean>>> inputs,
                Map<CPTCondition, DoubleVertex> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public DoubleCPTWhenBuilder when(Boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new DoubleCPTWhenBuilder(
                    new CPTCondition(ImmutableList.copyOf(condition)), this);
        }

        public DoubleCPTVertex orDefault(DoubleVertex defaultResult) {
            return new DoubleCPTVertex(inputs, conditions, defaultResult);
        }

        public DoubleCPTVertex orDefault(double defaultResult) {
            return orDefault(ConstantVertex.of(defaultResult));
        }

        public static class DoubleCPTWhenBuilder {

            private final CPTCondition condition;
            private final DoubleCPTBuilder builder;

            private DoubleCPTWhenBuilder(CPTCondition condition, DoubleCPTBuilder builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public DoubleCPTBuilder then(DoubleVertex thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }

            public DoubleCPTBuilder then(double thn) {
                return then(ConstantVertex.of(thn));
            }
        }
    }

    public static class CPTBuilder<T, OUT extends Tensor<T>> {
        private final List<Vertex<? extends Tensor<Boolean>>> inputs;
        private final Map<CPTCondition, Vertex<OUT>> conditions;

        public CPTBuilder(
                List<Vertex<? extends Tensor<Boolean>>> inputs,
                Map<CPTCondition, Vertex<OUT>> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public CPTWhenBuilder<T, OUT> when(Boolean... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenBuilder<>(new CPTCondition(ImmutableList.copyOf(condition)), this);
        }

        public CPTVertex<OUT> orDefault(Vertex<OUT> defaultResult) {
            return new CPTVertex<>(inputs, conditions, defaultResult);
        }

        public static class CPTWhenBuilder<T, OUT extends Tensor<T>> {

            private final CPTCondition condition;
            private final CPTBuilder<T, OUT> builder;

            private CPTWhenBuilder(CPTCondition condition, CPTBuilder<T, OUT> builder) {
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
