package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.Map;

@Builder
public class LogProbGraph {

    @Getter
    @Singular
    /*
     * parameter -> placeholder in log-prob-graph
     */
    private final Map<Vertex<?>, Vertex<?>> inputs;

    @Getter
    private final DoubleVertex logProbOutput;

    public <T> Vertex<T> getInput(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

}
