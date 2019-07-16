package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.BooleanVertexWrapper;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexWrapper;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;
import io.improbable.keanu.vertices.generic.GenericVertexWrapper;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertexWrapper;

public class If {

    private If() {
    }

    public static IfThenBuilder isTrue(BooleanVertex predicate) {
        return new IfThenBuilder(predicate);
    }

    public static class IfThenBuilder {
        private final BooleanVertex predicate;

        public IfThenBuilder(BooleanVertex predicate) {
            this.predicate = predicate;
        }

        public <T> IfThenElseBuilder<T> then(TensorVertex<T, GenericTensor<T>, ?> thn) {
            return new IfThenElseBuilder<>(predicate, thn);
        }

        public BooleanIfThenElseBuilder then(BooleanVertex thn) {
            return new BooleanIfThenElseBuilder(predicate, thn);
        }

        public BooleanIfThenElseBuilder then(boolean thn) {
            return then(ConstantVertex.of(thn));
        }

        public DoubleIfThenElseBuilder then(DoubleVertex thn) {
            return new DoubleIfThenElseBuilder(predicate, thn);
        }

        public DoubleIfThenElseBuilder then(double thn) {
            return then(ConstantVertex.of(thn));
        }

        public IntegerIfThenElseBuilder then(IntegerVertex thn) {
            return new IntegerIfThenElseBuilder(predicate, thn);
        }

        public IntegerIfThenElseBuilder then(int thn) {
            return then(ConstantVertex.of(thn));
        }
    }

    public static class IfThenElseBuilder<T> {

        private final TensorVertex<Boolean, BooleanTensor, ?> predicate;
        private final TensorVertex<T, GenericTensor<T>, ?> thn;

        public IfThenElseBuilder(TensorVertex<Boolean, BooleanTensor, BooleanVertex> predicate,
                                 TensorVertex<T, GenericTensor<T>, ?> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public GenericTensorVertex<T> orElse(TensorVertex<T, GenericTensor<T>, ?> els) {
            return new GenericVertexWrapper<>(new IfVertex<>(predicate, thn, els));
        }
    }

    public static class BooleanIfThenElseBuilder {

        private final TensorVertex<Boolean, BooleanTensor, ?> predicate;
        private final TensorVertex<Boolean, BooleanTensor, ?> thn;

        public BooleanIfThenElseBuilder(TensorVertex<Boolean, BooleanTensor, ?> predicate,
                                        TensorVertex<Boolean, BooleanTensor, ?> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public BooleanVertex orElse(TensorVertex<Boolean, BooleanTensor, ?> els) {
            return new BooleanVertexWrapper(new IfVertex<>(predicate, thn, els));
        }

        public BooleanVertex orElse(boolean els) {
            return orElse(ConstantVertex.of(els));
        }
    }

    public static class DoubleIfThenElseBuilder {

        private final TensorVertex<Boolean, BooleanTensor, ?> predicate;
        private final TensorVertex<Double, DoubleTensor, ?> thn;

        public DoubleIfThenElseBuilder(TensorVertex<Boolean, BooleanTensor, ?> predicate,
                                       TensorVertex<Double, DoubleTensor, ?> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public DoubleVertex orElse(TensorVertex<Double, DoubleTensor, ?> els) {
            return new DoubleVertexWrapper(new IfVertex<>(predicate, thn, els));
        }

        public DoubleVertex orElse(double els) {
            return orElse(ConstantVertex.of(els));
        }
    }

    public static class IntegerIfThenElseBuilder {

        private final TensorVertex<Boolean, BooleanTensor, ?> predicate;
        private final TensorVertex<Integer, IntegerTensor, ?> thn;

        public IntegerIfThenElseBuilder(TensorVertex<Boolean, BooleanTensor, ?> predicate,
                                        TensorVertex<Integer, IntegerTensor, ?> thn) {
            this.predicate = predicate;
            this.thn = thn;
        }

        public IntegerVertex orElse(TensorVertex<Integer, IntegerTensor, ?> els) {
            return new IntegerVertexWrapper(new IfVertex<>(predicate, thn, els));
        }

        public IntegerVertex orElse(int els) {
            return orElse(ConstantVertex.of(els));
        }
    }
}
