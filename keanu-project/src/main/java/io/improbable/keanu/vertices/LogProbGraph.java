package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
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

    /**
     * A vertex representing the result of log probability computation
     */
    @Getter
    private final DoubleVertex logProbOutput;

    public <T> Vertex<T> getPlaceHolder(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    static public class DoublePlaceHolderVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, Differentiable, NonSaveableVertex {

        public DoublePlaceHolderVertex(long[] initialShape) {
            super(initialShape);
        }

        @Override
        public DoubleTensor calculate() {
            return this.getValue();
        }

        @Override
        public DoubleTensor sample(KeanuRandom random) {
            return this.getValue();
        }
    }

    static public class IntegerPlaceHolderVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor>, Differentiable, NonSaveableVertex {

        public IntegerPlaceHolderVertex(long[] initialShape) {
            super(initialShape);
        }

        @Override
        public IntegerTensor calculate() {
            return this.getValue();
        }

        @Override
        public IntegerTensor sample(KeanuRandom random) {
            return this.getValue();
        }
    }
}
