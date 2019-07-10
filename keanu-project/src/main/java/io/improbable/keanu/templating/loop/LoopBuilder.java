package io.improbable.keanu.templating.loop;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.templating.SequenceItem;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public class LoopBuilder {
    private final VertexDictionary initialState;
    private ImmutableMap.Builder<VertexLabel, VertexLabel> customMappings = ImmutableMap.builder();
    private int maxLoopCount = Loop.DEFAULT_MAX_COUNT;
    private boolean throwWhenMaxCountIsReached = true;

    LoopBuilder(VertexDictionary initialState) {
        this.initialState = initialState;
    }

    /**
     * An optional method to override the default value
     *
     * @param maxCount the max number of times the loop can run
     * @return self
     */
    public LoopBuilder withMaxIterations(int maxCount) {
        this.maxLoopCount = maxCount;
        return this;
    }

    /**
     * An optional method to override the default behaviour
     * If the maximum loop count is exceeded, it will log a warning instead of throwing
     *
     * @return self
     */
    public LoopBuilder doNotThrowWhenMaxCountIsReached() {
        this.throwWhenMaxCountIsReached = false;
        return this;
    }

    /**
     * An optional method to add custom mappings
     *
     * @param proxyLabel        the label assigned to the ProxyVertex in frame t
     * @param proxysParentLabel the label assigned to a Vertex in frame t-1 which will become the ProxyVertex's parent
     * @return self
     */
    public LoopBuilder mapping(VertexLabel proxyLabel, VertexLabel proxysParentLabel) {
        customMappings.put(proxyLabel, proxysParentLabel);
        return this;
    }

    /**
     * A mandatory method to specify the condition
     *
     * @param conditionSupplier a lambda that creates and returns a new BooleanVertex
     * @return the next stage builder
     */
    public LoopBuilder2 iterateWhile(Supplier<BooleanVertex> conditionSupplier) {
        return iterateWhile(item -> conditionSupplier.get());
    }

    /**
     * A mandatory method to specify the condition
     *
     * @param conditionFunction a lambda that takes the current SequenceItem and creates and returns a new BooleanVertex
     * @return the next stage builder
     */
    public LoopBuilder2 iterateWhile(Function<SequenceItem, BooleanVertex> conditionFunction) {
        return new LoopBuilder2(initialState, conditionFunction, customMappings.build(), maxLoopCount, throwWhenMaxCountIsReached);
    }

    public class LoopBuilder2 {
        private final VertexDictionary initialState;
        private final Function<SequenceItem, BooleanVertex> conditionFunction;
        private final Map<VertexLabel, VertexLabel> customMappings;
        private final int maxLoopCount;
        private final boolean throwWhenMaxCountIsReached;
        private final VertexLabel VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL = new VertexLabel("loop_value_out_when_always_true");
        private final VertexLabel LOOP_LABEL = new VertexLabel("loop");


        LoopBuilder2(VertexDictionary initialState, Function<SequenceItem, BooleanVertex> conditionFunction, Map<VertexLabel, VertexLabel> customMappings, int maxLoopCount, boolean throwWhenMaxCountIsReached) {
            this.initialState = setInitialState(initialState);
            this.conditionFunction = conditionFunction;
            this.customMappings = customMappings;
            this.maxLoopCount = maxLoopCount;
            this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
        }

        private VertexDictionary setInitialState(VertexDictionary initialState) {
            Vertex valueOutWhenAlwaysTrue;

            try {
                Vertex<?, ?> outputVertex = initialState.get(Loop.VALUE_OUT_LABEL);
                valueOutWhenAlwaysTrue = new DoubleProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL.withExtraNamespace("Loop_" + this.hashCode()));
                valueOutWhenAlwaysTrue.setParents(outputVertex);
            } catch (NoSuchElementException e) {
                throw new LoopConstructionException("You must pass in a base case, i.e. a vertex labeled as Loop.VALUE_OUT_LABEL", e);
            } catch (IllegalArgumentException e) {
                throw new LoopConstructionException("You must pass in only one vertex labeled as Loop.VALUE_OUT_LABEL", e);
            }

            return initialState.withExtraEntries(ImmutableMap.of(
                VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL, valueOutWhenAlwaysTrue,
                LOOP_LABEL, ConstantVertex.of(true)));
        }

        /**
         * A mandatory method to specify the iteration step
         *
         * @param iterationFunction a lambda that takes the Proxy input vertex
         *                          and creates and returns a new output Vertex
         * @return the fully constructed Loop object
         */
        public Loop apply(Function<DoubleVertex, DoubleVertex> iterationFunction) {
            return apply((item, valueIn) -> iterationFunction.apply(valueIn));
        }

        /**
         * A mandatory method to specify the iteration step
         *
         * @param iterationFunction a lambda that takes the current SequenceItem and the Proxy input vertex
         *                          and creates and returns a new output vertex
         * @return the fully constructed Loop object
         */
        public Loop apply(BiFunction<SequenceItem, DoubleVertex, DoubleVertex> iterationFunction) {
            Sequence sequence = new SequenceBuilder<Integer>()
                .withInitialState(initialState)
                .withTransitionMapping(customMappings)
                .count(maxLoopCount)
                .withFactory((item) -> {
                    // inputs
                    DoubleVertex valueInWhenAlwaysTrue = item.addDoubleProxyFor(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
                    BooleanVertex stillLooping = item.addBooleanProxyFor(LOOP_LABEL);
                    DoubleVertex valueIn = item.addDoubleProxyFor(Loop.VALUE_OUT_LABEL);

                    // intermediate
                    BooleanVertex condition = conditionFunction.apply(item);
                    item.add(Loop.CONDITION_LABEL, condition);

                    // outputs
                    DoubleVertex iterationResult = iterationFunction.apply(item, valueInWhenAlwaysTrue);
                    BooleanVertex loopAgain = stillLooping.and(condition);
                    DoubleVertex result = If.isTrue(loopAgain).then(iterationResult).orElse(valueIn);
                    item.add(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL, iterationResult);
                    item.add(LOOP_LABEL, loopAgain);
                    item.add(Loop.VALUE_OUT_LABEL, result);
                })
                .build();

            return new Loop(sequence, throwWhenMaxCountIsReached);
        }
    }
}
