package io.improbable.keanu.plating.loop;

import java.util.Arrays;
import java.util.Collection;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateException;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;

public class Loop {
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    static final VertexLabel STILL_LOOPING = new VertexLabel("stillLooping");
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
        return new LoopBuilder(initialState);
    }

    public <V extends Vertex<? extends Tensor<?>>> V getOutput() {
        Plate finalPlate = plates.asList().get(plates.size() - 1);
        BoolVertex stillLooping = finalPlate.get(STILL_LOOPING);
        if (Arrays.stream(stillLooping.getValue().asFlatArray()).anyMatch(v -> v == true)) {
            throw new PlateException("Loop has exceeded its max count " + plates.size());
        }
        return finalPlate.get(VALUE_OUT_LABEL);
    }
}
