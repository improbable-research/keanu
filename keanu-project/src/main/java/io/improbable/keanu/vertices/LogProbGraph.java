package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
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

    public <T> Vertex<T> getInput(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    /**
     * A vertex representing the result of log probability computation
     */
    @Getter
    private final DoubleVertex logProbOutput;

    public <T> Vertex<T> getPlaceholder(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    static public class DoublePlaceholderVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, Differentiable, NonSaveableVertex {

        public DoublePlaceholderVertex(long... initialShape) {
            super(initialShape);
        }

        @Override
        public DoubleTensor calculate() {
            return getPlaceholderVertexValue(this);
        }

    }

    static public class IntegerPlaceholderVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor>, Differentiable, NonSaveableVertex {

        public IntegerPlaceholderVertex(long... initialShape) {
            super(initialShape);
        }

        @Override
        public IntegerTensor calculate() {
            return getPlaceholderVertexValue(this);
        }

    }

    static public class BooleanPlaceholderVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor>, Differentiable, NonSaveableVertex {

        public BooleanPlaceholderVertex(long... initialShape) {
            super(initialShape);
        }

        @Override
        public BooleanTensor calculate() {
            return getPlaceholderVertexValue(this);
        }

    }


    private static <T> T getPlaceholderVertexValue(Vertex<T> vertex) {
        if (!vertex.hasValue()) {
            throw new IllegalStateException("Cannot get value because PlaceholderVertex has not been initialized.");
        }
        return vertex.getValue();
    }
}
