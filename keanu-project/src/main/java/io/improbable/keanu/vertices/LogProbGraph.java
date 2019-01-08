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

    @Getter
    @Singular
    private final Map<Vertex<?>, Vertex<?>> inputs;

    @Getter
    private final DoubleVertex logProbOutput;

    public <T> Vertex<T> getInput(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    static public class PlaceHolderDoubleVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, Differentiable {

        public PlaceHolderDoubleVertex(long[] initialShape) {
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
