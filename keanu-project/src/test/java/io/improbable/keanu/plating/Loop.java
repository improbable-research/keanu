package io.improbable.keanu.plating;

import java.util.Collection;
import java.util.function.Function;
import java.util.function.Supplier;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class Loop {
    private Loop() {
    }

    public static LoopBuilder startingFrom(Vertex... initialState) {
        return new LoopBuilder(initialState);
    }

    public static class LoopBuilder {
        private final Collection<Vertex> initialState;

        LoopBuilder(Vertex... initialState) {
            this.initialState = ImmutableList.copyOf(initialState);
        }

        public LoopBuilder2 apply(Function<DoubleVertex, DoubleVertex> condition) {
            return new LoopBuilder2(initialState, condition);
        }

        public class LoopBuilder2 {
            private final Function<DoubleVertex, DoubleVertex> iterationFunction;
            private final Collection<Vertex> initialState;


            LoopBuilder2(Collection<Vertex> initialState, Function<DoubleVertex, DoubleVertex> iterationFunction) {
                this.initialState = initialState;
                this.iterationFunction = iterationFunction;
            }

            public BayesianNetwork whilst(Supplier<BoolVertex> conditionSupplier) throws VertexLabelException {
                VertexLabel lambdaInLabel = createVertexLabel("lambdaIn");
                VertexLabel lambdaOutLabel = createVertexLabel("lambdaOut");
                VertexLabel loopLabel = createVertexLabel("loop");
                VertexLabel stillLoopingLabel = createVertexLabel("stillLooping");
                VertexLabel valueInLabel = createVertexLabel("valueIn");
                VertexLabel valueOutLabel = createVertexLabel("valueOut");

                Plates plates = new PlateBuilder<Integer>()
                    .withInitialState(initialState.toArray(new Vertex[0]))
                    .withProxyMapping(ImmutableMap.of(
                        lambdaInLabel, lambdaOutLabel,
                        stillLoopingLabel, loopLabel,
                        valueInLabel, valueOutLabel
                    ))
                    .count(100)
                    .withFactory((plate) -> {
                        // inputs
                        DoubleVertex runningTotal = new DoubleProxyVertex(lambdaInLabel);
                        BoolVertex stillLooping = new BoolProxyVertex(stillLoopingLabel);
                        DoubleVertex valueIn = new DoubleProxyVertex(valueInLabel);
                        plate.addAll(ImmutableSet.of(runningTotal, stillLooping, valueIn));

                        // intermediate
                        BoolVertex condition = conditionSupplier.get();
                        plate.add(condition);

                        // outputs
                        DoubleVertex plus = iterationFunction.apply(valueIn);
                        BoolVertex loopAgain = stillLooping.and(condition).labelled(loopLabel);
                        DoubleVertex result = If.isTrue(loopAgain).then(plus).orElse(valueIn).labelled(valueOutLabel);
                        plate.addAll(ImmutableSet.of(plus, loopAgain, result));
                    })
                    .build();

                return new BayesianNetwork(initialState.iterator().next().getConnectedGraph()); // TODO cleaner?
            }

            private VertexLabel createVertexLabel(String loop) {
                return new VertexLabel(loop, getId());
            }

            private String getId() {
                return "Plates_" + hashCode();
            }

        }
    }
}
