package io.improbable.keanu.plating.loop;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;

import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.PlateException;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class LoopBuilder {
    private int maxLoopCount = Loop.DEFAULT_MAX_COUNT;
    private final Collection<Vertex> initialState;
    private final ImmutableMap.Builder<VertexLabel, VertexLabel> customMappings = ImmutableMap.builder();
    private boolean throwWhenMaxCountIsReached = true;


    <V extends Vertex<?>> LoopBuilder(Collection<V> initialState) {
        this.initialState = ImmutableList.copyOf(initialState);
    }

    public LoopBuilder atMost(int maxCount) {
        this.maxLoopCount = maxCount;
        return this;
    }

    public LoopBuilder dontThrowWhenMaxCountIsReached() {
        this.throwWhenMaxCountIsReached = false;
        return this;
    }

    public LoopBuilder mapping(VertexLabel inputLabel, VertexLabel outputLabel) {
        customMappings.put(inputLabel, outputLabel);
        return this;
    }

    public LoopBuilder2 whilst(Supplier<BoolVertex> conditionSupplier) throws VertexLabelException {
        return whilst(plate -> conditionSupplier.get());
    }

    public LoopBuilder2 whilst(Function<Plate, BoolVertex> conditionFunction) throws VertexLabelException {
        return new LoopBuilder2(maxLoopCount, initialState, conditionFunction, throwWhenMaxCountIsReached, customMappings.build());
    }

    public class LoopBuilder2 {
        private final Function<Plate, BoolVertex> conditionFunction;
        private final boolean throwWhenMaxCountIsReached;
        private final Map<VertexLabel, VertexLabel> customMappings;
        private final int maxLoopCount;
        private final Collection<Vertex> initialState;
        private final VertexLabel VALUE_IN_WHEN_ALWAYS_TRUE_LABEL = new VertexLabel("loop_value_in_when_always_true");
        private final VertexLabel VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL = new VertexLabel("loop_value_out_when_always_true");
        private final VertexLabel LOOP_LABEL = new VertexLabel("loop");


        LoopBuilder2(int maxLoopCount, Collection<Vertex> initialState, Function<Plate, BoolVertex> conditionFunction, boolean throwWhenMaxCountIsReached, Map<VertexLabel, VertexLabel> customMappings) {
            this.maxLoopCount = maxLoopCount;
            this.initialState = setInitialState(initialState);
            this.conditionFunction = conditionFunction;
            this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
            this.customMappings = customMappings;
        }

        private ImmutableList<Vertex> setInitialState(Collection<Vertex> initialState) {
            Vertex valueOutWhenAlwaysTrue = null;

            try {
                List<Vertex> outputVertices = initialState.stream().filter(v -> Loop.VALUE_OUT_LABEL.equals(v.getLabel())).collect(Collectors.toList());
                Vertex<?> outputVertex = Iterables.getOnlyElement(outputVertices);
                valueOutWhenAlwaysTrue = new DoubleProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
                valueOutWhenAlwaysTrue.setParents(outputVertex);
            } catch (NoSuchElementException e) {
                throw new PlateException("You must pass in a base case, i.e. a vertex labelled with Loop.VALUE_OUT_LABEL", e);
            }

            BoolVertex tru = ConstantVertex.of(true).labelled(LOOP_LABEL);

            return ImmutableList.<Vertex>builder()
                .addAll(initialState)
                .add(valueOutWhenAlwaysTrue)
                .add(tru)
                .build();
        }


        public Loop apply(Function<DoubleVertex, DoubleVertex> iterationFunction) throws VertexLabelException {
            return apply((plate, valueIn) -> {
                return iterationFunction.apply(valueIn);
            });
        }

        public Loop apply(BiFunction<Plate, DoubleVertex, DoubleVertex> iterationFunction) throws VertexLabelException {
            Plates plates = new PlateBuilder<Integer>()
                .withInitialState(initialState.toArray(new Vertex[0]))
                .withProxyMapping(ImmutableMap.<VertexLabel, VertexLabel>builder()
                    .putAll(customMappings)
                    .put(VALUE_IN_WHEN_ALWAYS_TRUE_LABEL, VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL)
                    .put(Loop.STILL_LOOPING, LOOP_LABEL)
                    .put(Loop.VALUE_IN_LABEL, Loop.VALUE_OUT_LABEL)
                    .build()
                )
                .count(maxLoopCount)
                .withFactory((plate) -> {
                    // inputs
                    DoubleVertex valueInWhenAlwaysTrue = new DoubleProxyVertex(VALUE_IN_WHEN_ALWAYS_TRUE_LABEL);
                    BoolVertex stillLooping = new BoolProxyVertex(Loop.STILL_LOOPING);
                    DoubleVertex valueIn = new DoubleProxyVertex(Loop.VALUE_IN_LABEL);
                    plate.addAll(ImmutableSet.of(valueInWhenAlwaysTrue, stillLooping, valueIn));

                    // intermediate
                    BoolVertex condition = conditionFunction.apply(plate).labelled(Loop.CONDITION_LABEL);
                    plate.add(condition);

                    // outputs
                    DoubleVertex iterationResult = iterationFunction.apply(plate, valueInWhenAlwaysTrue).labelled(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
                    BoolVertex loopAgain = stillLooping.and(condition).labelled(LOOP_LABEL);
                    DoubleVertex result = If.isTrue(loopAgain).then(iterationResult).orElse(valueIn).labelled(Loop.VALUE_OUT_LABEL);
                    plate.addAll(ImmutableSet.of(iterationResult, loopAgain, result));
                })
                .build();

            return new Loop(plates, throwWhenMaxCountIsReached);
        }

    }
}
