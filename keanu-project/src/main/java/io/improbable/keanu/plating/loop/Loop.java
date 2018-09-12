package io.improbable.keanu.plating.loop;

import java.util.Arrays;
import java.util.Collection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateException;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;

public class Loop {
    private Logger log = LoggerFactory.getLogger(this.getClass());
    public static final VertexLabel VALUE_IN_LABEL = new VertexLabel("loop_value_in");
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    public static final VertexLabel CONDITION_LABEL = new VertexLabel("loop_condition");
    static final VertexLabel STILL_LOOPING = new VertexLabel("stillLooping");
    public static final int DEFAULT_MAX_COUNT = 100;
    private final Plates plates;
    private final boolean throwWhenMaxCountIsReached;

    /**
     * package-private because it is intended to be created by the LoopBuilder
     *
     * @param plates
     * @param throwWhenMaxCountIsReached
     */
    Loop(Plates plates, boolean throwWhenMaxCountIsReached) {
        this.plates = plates;
        this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
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
        checkIfMaxCountHasBeenReached(finalPlate);
        return finalPlate.get(VALUE_OUT_LABEL);
    }

    private void checkIfMaxCountHasBeenReached(Plate plate) {
        BoolVertex stillLooping = plate.get(STILL_LOOPING);
        if (Arrays.stream(stillLooping.getValue().asFlatArray()).anyMatch(v -> v == true)) {
            String errorMessage = "Loop has exceeded its max count " + plates.size();
            if (throwWhenMaxCountIsReached) {
                throw new PlateException(errorMessage);
            } else {
                log.warn(errorMessage);
            }
        }
    }
}
