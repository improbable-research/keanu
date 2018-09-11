package io.improbable.keanu.plating.loop;

import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

public class Loop {
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    private final Plates plates;
    private final int maximumLoopLength;

    /**
     * package-private because it is intended to be created by the LoopBuilder
     * @param plates
     * @param maximumLoopLength
     */
    Loop(Plates plates, int maximumLoopLength) {
        this.plates = plates;
        this.maximumLoopLength = maximumLoopLength;
    }

    public static LoopBuilder startingFrom(Vertex... initialState) {
        return new LoopBuilder(initialState);
    }

    public <V extends Vertex<? extends Tensor<?>>> V getOutput() {
        return plates.asList().get(maximumLoopLength - 1).get(VALUE_OUT_LABEL);
    }
}
