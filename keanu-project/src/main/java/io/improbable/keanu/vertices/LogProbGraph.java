package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.Set;

@Builder
public class LogProbGraph {

    @Getter
    @Singular
    private final Set<ProxyVertex<?>> params;

    @Getter
    private final Vertex<?> x;

    @Getter
    private final DoubleVertex logProbOutput;

    public <T> void setXValue(T value) {
        ((Vertex <T>) x).setValue(value);
    }

    static public class DoublePlaceHolderVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, Differentiable {

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
