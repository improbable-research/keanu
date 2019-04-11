package io.improbable.keanu.vertices.generic.nonprobabilistic;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleCPTVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerCPTVertex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ConditionalProbabilityTable {

    private ConditionalProbabilityTable() {
    }

    private static final String WHEN_CONDITION_SIZE_MISMATCH = "The 'when' condition size does not match input count";

    @SafeVarargs
    public static <IN> CPTRawBuilder<IN> of(Vertex<? extends Tensor<IN>>... inputs) {
        return new CPTRawBuilder<>(Arrays.asList(inputs));
    }

    public static class CPTRawBuilder<IN> {

        private final List<Vertex<? extends Tensor<IN>>> inputs;

        public CPTRawBuilder(List<Vertex<? extends Tensor<IN>>> inputs) {
            this.inputs = inputs;
        }

        @SafeVarargs
        public final CPTWhenRawBuilder<IN> when(IN... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenRawBuilder<>(ImmutableList.copyOf(condition), inputs);
        }
    }

    public static class CPTWhenRawBuilder<IN> {

        private final CPTCondition<IN> condition;
        private final List<Vertex<? extends Tensor<IN>>> inputs;

        public CPTWhenRawBuilder(List<IN> condition, List<Vertex<? extends Tensor<IN>>> inputs) {
            this.condition = new CPTCondition<>(condition);
            this.inputs = inputs;
        }

        public <OUT extends Tensor> CPTBuilder<IN, OUT> then(Vertex<OUT> thn) {
            Map<CPTCondition<IN>, Vertex<OUT>> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new CPTBuilder<>(inputs, conditions);
        }

        public DoubleCPTBuilder<IN> then(DoubleVertex thn) {
            Map<CPTCondition<IN>, DoubleVertex> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new DoubleCPTBuilder<>(inputs, conditions);
        }

        public DoubleCPTBuilder<IN> then(double thn) {
            return then(ConstantVertex.of(thn));
        }

        public IntegerCPTBuilder<IN> then(IntegerVertex thn) {
            Map<CPTCondition<IN>, IntegerVertex> conditions = new HashMap<>();
            conditions.put(condition, thn);
            return new IntegerCPTBuilder<>(inputs, conditions);
        }

        public IntegerCPTBuilder<IN> then(int thn) {
            return then(ConstantVertex.of(thn));
        }
    }

    public static class DoubleCPTBuilder<IN> {
        private final List<Vertex<? extends Tensor<IN>>> inputs;
        private final Map<CPTCondition<IN>, DoubleVertex> conditions;

        public DoubleCPTBuilder(List<Vertex<? extends Tensor<IN>>> inputs, Map<CPTCondition<IN>, DoubleVertex> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public DoubleCPTWhenBuilder<IN> when(IN... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new DoubleCPTWhenBuilder<>(new CPTCondition<>(ImmutableList.copyOf(condition)), this);
        }

        public DoubleCPTVertex<IN> orDefault(DoubleVertex defaultResult) {
            return new DoubleCPTVertex<>(inputs, conditions, defaultResult);
        }

        public DoubleCPTVertex<IN> orDefault(double defaultResult) {
            return orDefault(ConstantVertex.of(defaultResult));
        }

        public static class DoubleCPTWhenBuilder<IN> {

            private final CPTCondition<IN> condition;
            private final DoubleCPTBuilder<IN> builder;

            private DoubleCPTWhenBuilder(CPTCondition<IN> condition, DoubleCPTBuilder<IN> builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public DoubleCPTBuilder<IN> then(DoubleVertex thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }

            public DoubleCPTBuilder<IN> then(double thn) {
                return then(ConstantVertex.of(thn));
            }
        }
    }

    public static class IntegerCPTBuilder<IN> {
        private final List<Vertex<? extends Tensor<IN>>> inputs;
        private final Map<CPTCondition<IN>, IntegerVertex> conditions;

        public IntegerCPTBuilder(List<Vertex<? extends Tensor<IN>>> inputs, Map<CPTCondition<IN>, IntegerVertex> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public IntegerCPTWhenBuilder<IN> when(IN... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new IntegerCPTWhenBuilder<>(new CPTCondition<>(ImmutableList.copyOf(condition)), this);
        }

        public IntegerCPTVertex<IN> orDefault(IntegerVertex defaultResult) {
            return new IntegerCPTVertex<>(inputs, conditions, defaultResult);
        }

        public IntegerCPTVertex<IN> orDefault(int defaultResult) {
            return orDefault(ConstantVertex.of(defaultResult));
        }

        public static class IntegerCPTWhenBuilder<IN> {

            private final CPTCondition<IN> condition;
            private final IntegerCPTBuilder<IN> builder;

            private IntegerCPTWhenBuilder(CPTCondition<IN> condition, IntegerCPTBuilder<IN> builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public IntegerCPTBuilder<IN> then(IntegerVertex thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }

            public IntegerCPTBuilder<IN> then(int thn) {
                return then(ConstantVertex.of(thn));
            }
        }
    }

    public static class CPTBuilder<IN, OUT extends Tensor> {
        private final List<Vertex<? extends Tensor<IN>>> inputs;
        private final Map<CPTCondition<IN>, Vertex<OUT>> conditions;

        public CPTBuilder(List<Vertex<? extends Tensor<IN>>> inputs, Map<CPTCondition<IN>, Vertex<OUT>> conditions) {
            this.inputs = inputs;
            this.conditions = conditions;
        }

        public CPTWhenBuilder<IN, OUT> when(IN... condition) {
            if (condition.length != inputs.size()) {
                throw new IllegalArgumentException(WHEN_CONDITION_SIZE_MISMATCH);
            }
            return new CPTWhenBuilder<>(new CPTCondition<>(ImmutableList.copyOf(condition)), this);
        }

        public CPTVertex<IN, OUT> orDefault(Vertex<OUT> defaultResult) {
            return new CPTVertex<>(inputs, conditions, defaultResult);
        }

        public static class CPTWhenBuilder<IN, OUT extends Tensor> {

            private final CPTCondition<IN> condition;
            private final CPTBuilder<IN, OUT> builder;

            private CPTWhenBuilder(CPTCondition<IN> condition, CPTBuilder<IN, OUT> builder) {
                this.condition = condition;
                this.builder = builder;
            }

            public CPTBuilder<IN, OUT> then(Vertex<OUT> thn) {
                builder.conditions.put(condition, thn);
                return builder;
            }
        }
    }
}
