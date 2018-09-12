package io.improbable.keanu.plating.loop;

import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;
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
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;

public class LoopBuilder {
    private int maxLoopCount = Loop.DEFAULT_MAX_COUNT;
    private final Collection<Vertex> initialState;
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

    public LoopBuilder2 apply(Function<DoubleVertex, DoubleVertex> condition) {
        return new LoopBuilder2(maxLoopCount, initialState, condition, throwWhenMaxCountIsReached);
    }

    public class LoopBuilder2 {
        private final Function<DoubleVertex, DoubleVertex> iterationFunction;
        private final boolean throwWhenMaxCountIsReached;
        private final int maxLoopCount;
        private final Collection<Vertex> initialState;
        private final VertexLabel VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL = new VertexLabel("loop_value_out_when_always_true");
        private final VertexLabel LOOP_LABEL = new VertexLabel("loop");


        LoopBuilder2(int maxLoopCount, Collection<Vertex> initialState, Function<DoubleVertex, DoubleVertex> iterationFunction, boolean throwWhenMaxCountIsReached) {
            this.maxLoopCount = maxLoopCount;
            this.initialState = setInitialState(initialState);
            this.iterationFunction = iterationFunction;
            this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
        }

        private ImmutableList<Vertex> setInitialState(Collection<Vertex> initialState) {
            Vertex valueOutWhenAlwaysTrue = null;

            try {
                List<Vertex> outputVertices = initialState.stream().filter(v -> v.getLabel().equals(Loop.VALUE_OUT_LABEL)).collect(Collectors.toList());
                Vertex outputVertex = Iterables.getOnlyElement(outputVertices);
                valueOutWhenAlwaysTrue = createProxyFor(outputVertex);
            } catch(NoSuchElementException | NullPointerException e) {
                throw new PlateException("You must pass in a base case, i.e. a vertex labelled with Loop.VALUE_OUT_LABEL", e);
            }

            BoolVertex tru = ConstantVertex.of(true).labelled(LOOP_LABEL);

            return ImmutableList.<Vertex>builder()
                .addAll(initialState)
                .add(valueOutWhenAlwaysTrue)
                .add(tru)
                .build();
        }

        private <V extends Vertex<?>> V createProxyFor(Vertex vertex) {
            V proxy;
            if (vertex instanceof DoubleVertex) {
                proxy = (V) new DoubleProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
            } else if (vertex instanceof IntegerVertex) {
                proxy = (V) new IntegerProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
            } else if (vertex instanceof BoolVertex) {
                proxy = (V) new BoolProxyVertex(VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL);
            } else {
                throw new PlateException("Input Vertex must be of type DoubleVertex, IntegerVertex or BoolVertex");
            }
            proxy.setParents(vertex);
            return proxy;
        }

        public Loop whilst(Supplier<BoolVertex> conditionSupplier) throws VertexLabelException {
            return whilst(plate -> conditionSupplier.get());
        }

        public Loop whilst(Function<Plate,BoolVertex> conditionFunction) throws VertexLabelException {
            // inputs
            VertexLabel valueInWhenAlwaysTrueLabel = new VertexLabel("valueInWhenAlwaysTrue");
            VertexLabel stillLoopingLabel = Loop.STILL_LOOPING;
            VertexLabel valueInLabel = Loop.VALUE_IN_LABEL;

            // intermediate
            VertexLabel conditionLabel = new VertexLabel("condition");

            // outputs
            VertexLabel valueOutWhenAlwaysTrueLabel = VALUE_OUT_WHEN_ALWAYS_TRUE_LABEL;
            VertexLabel loopLabel = LOOP_LABEL;
            VertexLabel valueOutLabel = Loop.VALUE_OUT_LABEL;

            Plates plates = new PlateBuilder<Integer>()
                .withInitialState(initialState.toArray(new Vertex[0]))
                .withProxyMapping(ImmutableMap.of(
                    valueInWhenAlwaysTrueLabel, valueOutWhenAlwaysTrueLabel,
                    stillLoopingLabel, loopLabel,
                    valueInLabel, valueOutLabel
                ))
                .count(maxLoopCount)
                .withFactory((plate) -> {
                    // inputs
                    DoubleVertex valueInWhenAlwaysTrue = new DoubleProxyVertex(valueInWhenAlwaysTrueLabel);
                    BoolVertex stillLooping = new BoolProxyVertex(stillLoopingLabel);
                    DoubleVertex valueIn = new DoubleProxyVertex(valueInLabel);
                    plate.addAll(ImmutableSet.of(valueInWhenAlwaysTrue, stillLooping, valueIn));

                    // intermediate
                    BoolVertex condition = conditionFunction.apply(plate).labelled(conditionLabel);
                    plate.add(condition);

                    // outputs
                    DoubleVertex iterationResult = iterationFunction.apply(valueInWhenAlwaysTrue).labelled(valueOutWhenAlwaysTrueLabel);
                    BoolVertex loopAgain = stillLooping.and(condition).labelled(loopLabel);
                    DoubleVertex result = If.isTrue(loopAgain).then(iterationResult).orElse(valueIn).labelled(valueOutLabel);
                    plate.addAll(ImmutableSet.of(iterationResult, loopAgain, result));
                })
                .build();

            return new Loop(plates, throwWhenMaxCountIsReached);
        }

    }
}
