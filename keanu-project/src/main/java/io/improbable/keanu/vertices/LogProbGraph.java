package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
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

    public static class DoublePlaceholderVertex extends DoubleVertex implements PlaceholderVertex, NonProbabilistic<DoubleTensor>, Differentiable {

        public DoublePlaceholderVertex(@LoadShape long[] initialShape) {
            super(initialShape);
        }

        @Override
        public DoubleTensor calculate() {
            return getPlaceholderVertexValue(this);
        }

        @Override
        public DoubleTensor sample(KeanuRandom random) {
            return getPlaceholderVertexValue(this);
        }
    }

    public static class PlaceHolderBoolVertex extends BooleanVertex implements PlaceholderVertex, NonProbabilistic<BooleanTensor> {

        public PlaceHolderBoolVertex(@LoadShape long[] initialShape) {
            super(initialShape);
        }

        @Override
        public BooleanTensor calculate() {
            return this.getValue();
        }

        @Override
        public BooleanTensor sample(KeanuRandom random) {
            return this.getValue();
        }
    }

    public interface PlaceholderVertex {
    }

    private static <T> T getPlaceholderVertexValue(Vertex<T> vertex) {
        if (!vertex.hasValue()) {
            throw new IllegalStateException("Cannot get value because PlaceholderVertex has not been initialized.");
        }
        return vertex.getValue();
    }
}
