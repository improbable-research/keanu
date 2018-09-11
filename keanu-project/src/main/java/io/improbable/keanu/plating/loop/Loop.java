package io.improbable.keanu.plating.loop;

import java.util.Collection;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

public class Loop {
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    public static final int DEFAULT_MAX_COUNT = 100;
    private final Plates plates;

    /**
     * package-private because it is intended to be created by the LoopBuilder
     * @param plates
     *
     */
    Loop(Plates plates) {
        this.plates = plates;
    }

    public Plates getPlates() {
        return plates;
    }

    public static <V extends Vertex<?>> LoopBuilder startingFrom(V initialState) {
        if (initialState.getLabel() == null) {
            initialState.setLabel(VALUE_OUT_LABEL);
        }
        return startingFrom(ImmutableList.of(initialState));
    }

    public static <V extends Vertex<?>> LoopBuilder startingFrom(V first, V... others) {
        return startingFrom(ImmutableList.<V>builder().add(first).add(others).build());
    }

    public static <V extends Vertex<?>> LoopBuilder startingFrom(Collection<V> initialState) {
        return new LoopBuilder(DEFAULT_MAX_COUNT, initialState);
    }

    public <V extends Vertex<? extends Tensor<?>>> V getOutput() {
        return plates.asList().get(plates.size() - 1).get(VALUE_OUT_LABEL);
    }
}
