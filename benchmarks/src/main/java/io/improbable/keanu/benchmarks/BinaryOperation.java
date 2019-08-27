package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public enum BinaryOperation {
    PLUS {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.plus(rhs);
        }
    },
    MINUS {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.minus(rhs);
        }
    },
    TIMES {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.times(rhs);
        }
    },
    DIVIDE {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.div(rhs);
        }
    },
    MATRIX_MULTIPLY {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.matrixMultiply(rhs);
        }
    };

    public abstract DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs);
}
