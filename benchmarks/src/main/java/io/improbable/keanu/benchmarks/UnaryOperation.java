package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public enum UnaryOperation {

    COS {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.cos();
        }
    },

    DETERMINANT {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.matrixDeterminant();
        }
    },

    CHOLESKY {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.choleskyDecomposition();
        }
    },

    INVERSE {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.matrixInverse();
        }
    },

    LOG {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.log();
        }
    },

    TRANSPOSE {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.transpose();
        }
    },

    SUM_DIM1 {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.sum(1);
        }
    },

    SLICE_DIM1_INDEX1 {
        @Override
        public DoubleTensor apply(DoubleTensor tensor) {
            return tensor.slice(1, 1);
        }
    };

    public abstract DoubleTensor apply(DoubleTensor tensor);
}
