package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.Map;

/**
 * A graph of vertices representing the computation of a log probability for a specific random variable
 */
@Builder
public class LogProbGraph {

    /**
     * A mapping from vertices to placeholders. The two are not explicitly linked together
     * to avoid mutating the vertex's existing network.
     */
    @Getter
    @Singular
    private final Map<Vertex<?>, Vertex<?>> inputs;

    /**
     * A vertex representing the result of log probability computation
     */
    @Getter
    private final DoubleVertex logProbOutput;

    public <T> Vertex<T> getPlaceholder(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    public interface PlaceholderVertex {
    }
}
