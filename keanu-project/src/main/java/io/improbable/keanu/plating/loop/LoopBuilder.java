package io.improbable.keanu.plating.loop;

import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import com.google.common.collect.ImmutableMap;

import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class LoopBuilder {
    private final VertexDictionary initialState;
    private ImmutableMap.Builder<VertexLabel, VertexLabel> customMappings = ImmutableMap.builder();
    private int maxLoopCount = Loop.DEFAULT_MAX_COUNT;
    private boolean throwWhenMaxCountIsReached = true;

    <V extends Vertex<?>> LoopBuilder(VertexDictionary initialState) {
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
     * @param inputLabel  the label assigned to the ProxyVertex in frame t
     * @param outputLabel the label assigned to a Vertex in frame t-1 which will become the ProxyVertex's parent
     * @return self
     */
    public LoopBuilder mapping(VertexLabel inputLabel, VertexLabel outputLabel) {
        customMappings.put(inputLabel, outputLabel);
        return this;
    }

    /**
     * A mandatory method to specify the condition
     *
     * @param conditionSupplier a lambda that creates and returns a new BoolVertex
     * @return the next stage builder
     */
    public LoopBuilder2 whilst(Supplier<BoolVertex> conditionSupplier) {
        return whilst(plate -> conditionSupplier.get());
    }

    /**
     * A mandatory method to specify the condition
     *
     * @param conditionFunction a lambda that takes the current Plate and creates and returns a new BoolVertex
     * @return the next stage builder
     */
    public LoopBuilder2 whilst(Function<Plate, BoolVertex> conditionFunction) {
        return new LoopBuilder2(initialState, conditionFunction, customMappings.build(), maxLoopCount, throwWhenMaxCountIsReached);
    }

    public class LoopBuilder2 {
        private final VertexDictionary initialState;
        private final Function<Plate, BoolVertex> conditionFunction;
        private final Map<VertexLabel, VertexLabel> customMappings;
        private final int maxLoopCount;
        private final boolean throwWhenMaxCountIsReached;
        private final VertexLabel VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL = new VertexLabel("loop_value_out_when_always_true");
        private final VertexLabel VALUE_IN_WHEN_ALWAYS_TRUE_LABEL = PlateBuilder.proxyFor(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
        private final VertexLabel LOOP_LABEL = new VertexLabel("loop");


        LoopBuilder2(VertexDictionary initialState, Function<Plate, BoolVertex> conditionFunction, Map<VertexLabel, VertexLabel> customMappings, int maxLoopCount, boolean throwWhenMaxCountIsReached) {
            this.initialState = setInitialState(initialState);
            this.conditionFunction = conditionFunction;
            this.customMappings = customMappings;
            this.maxLoopCount = maxLoopCount;
            this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
        }

        private VertexDictionary setInitialState(VertexDictionary initialState) {
            Vertex valueOutWhenAlwaysTrue;

            try {
                Vertex<?> outputVertex = initialState.get(Loop.VALUE_OUT_LABEL);
                valueOutWhenAlwaysTrue = new DoubleProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL.withExtraNamespace("Loop_" + this.hashCode()));
                valueOutWhenAlwaysTrue.setParents(outputVertex);
            } catch (NoSuchElementException e) {
                throw new VertexLabelException("You must pass in a base case, i.e. a vertex labeled as Loop.VALUE_OUT_LABEL", e);
            } catch (IllegalArgumentException e) {
                throw new VertexLabelException("You must pass in only one vertex labeled as Loop.VALUE_OUT_LABEL", e);
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
            return apply((plate, valueIn) -> iterationFunction.apply(valueIn));
        }

        /**
         * A mandatory method to specify the iteration step
         *
         * @param iterationFunction a lambda that takes the current Plate and the Proxy input vertex
         *                          and creates and returns a new output vertex
         * @return the fully constructed Loop object
         */
        public Loop apply(BiFunction<Plate, DoubleVertex, DoubleVertex> iterationFunction) {
            Plates plates = new PlateBuilder<Integer>()
                .withInitialState(initialState)
                .withTransitionMapping(customMappings)
                .count(maxLoopCount)
                .withFactory((plate) -> {
                    // inputs
                    DoubleVertex valueInWhenAlwaysTrue = new DoubleProxyVertex(VALUE_IN_WHEN_ALWAYS_TRUE_LABEL);
                    BoolVertex stillLooping = new BoolProxyVertex(Loop.STILL_LOOPING_LABEL);
                    DoubleVertex valueIn = new DoubleProxyVertex(Loop.VALUE_IN_LABEL);
                    plate.addAll(valueInWhenAlwaysTrue, stillLooping, valueIn);

                    // intermediate
                    BoolVertex condition = conditionFunction.apply(plate);
                    plate.add(Loop.CONDITION_LABEL, condition);

                    // outputs
                    DoubleVertex iterationResult = iterationFunction.apply(plate, valueInWhenAlwaysTrue).labeledAs(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
                    BoolVertex loopAgain = stillLooping.and(condition).labeledAs(LOOP_LABEL);
                    DoubleVertex result = If.isTrue(loopAgain).then(iterationResult).orElse(valueIn).labeledAs(Loop.VALUE_OUT_LABEL);
                    plate.add(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL, iterationResult);
                    plate.add(LOOP_LABEL, loopAgain);
                    plate.add(Loop.VALUE_OUT_LABEL, result);
                })
                .build();

            return new Loop(plates, throwWhenMaxCountIsReached);
        }
    }
}
