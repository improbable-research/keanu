package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanIfVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

import java.util.Arrays;

public class If {

    private If() {
    }

    public static IfThenBuilder isTrue(Vertex<BooleanTensor> predicate) {
        return new IfThenBuilder(predicate);
    }

    public static class IfThenBuilder {
        private final Vertex<? extends BooleanTensor> predicate;

        public IfThenBuilder(Vertex<? extends BooleanTensor> predicate) {
            this.predicate = predicate;
        }

        public <T> IfThenElseBuilder<T> then(Vertex<? extends Tensor<T>> thn) {
            return new IfThenElseBuilder<>(predicate, thn);
        }

        public BooleanIfThenElseBuilder then(BoolVertex thn) {
            return new BooleanIfThenElseBuilder(predicate, thn);
        }

        public BooleanIfThenElseBuilder then(boolean thn) {
            return then(new ConstantBoolVertex(thn));
        }

        public DoubleIfThenElseBuilder then(DoubleVertex thn) {
            return new DoubleIfThenElseBuilder(predicate, thn);
        }

        public DoubleIfThenElseBuilder then(double thn) {
            return then(new ConstantDoubleVertex(thn));
        }
    }

    public static class IfThenElseBuilder<T> {

        private final Vertex<? extends BooleanTensor> predicate;
        private final Vertex<? extends Tensor<T>> thn;

        public IfThenElseBuilder(Vertex<? extends BooleanTensor> predicate,
                                 Vertex<? extends Tensor<T>> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public Vertex<Tensor<T>> orElse(Vertex<? extends Tensor<T>> els) {
            if (Arrays.equals(thn.getShape(), els.getShape())) {
                return new IfVertex<>(els.getShape(), predicate, thn, els);
            } else {
                throw new IllegalArgumentException("Else must match then shape");
            }
        }
    }

    public static class BooleanIfThenElseBuilder {

        private final Vertex<? extends BooleanTensor> predicate;
        private final Vertex<? extends BooleanTensor> thn;

        public BooleanIfThenElseBuilder(Vertex<? extends BooleanTensor> predicate,
                                        Vertex<? extends BooleanTensor> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public BoolVertex orElse(Vertex<? extends BooleanTensor> els) {
            if (Arrays.equals(thn.getShape(), els.getShape())) {
                return new BooleanIfVertex(els.getShape(), predicate, thn, els);
            } else {
                throw new IllegalArgumentException("Else must match then shape");
            }
        }
    }

    public static class DoubleIfThenElseBuilder {

        private final Vertex<? extends BooleanTensor> predicate;
        private final Vertex<? extends DoubleTensor> thn;

        public DoubleIfThenElseBuilder(Vertex<? extends BooleanTensor> predicate,
                                       Vertex<? extends DoubleTensor> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public DoubleVertex orElse(Vertex<? extends DoubleTensor> els) {
            if (Arrays.equals(thn.getShape(), els.getShape())) {
                return new DoubleIfVertex(els.getShape(), predicate, thn, els);
            } else {
                throw new IllegalArgumentException("Else must match then shape");
            }
        }

        public DoubleVertex orElse(double els) {
            return orElse(new ConstantDoubleVertex(els));
        }
    }
}
