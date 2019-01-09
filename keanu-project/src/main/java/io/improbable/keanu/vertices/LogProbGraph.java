package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.Map;

@Builder
public class LogProbGraph {

    /**
     * Mapping from Keanu vertices to placeholders, which would have its values fed during execution
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
}
