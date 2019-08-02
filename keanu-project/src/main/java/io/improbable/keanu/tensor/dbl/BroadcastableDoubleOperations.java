package io.improbable.keanu.tensor.dbl;

import org.apache.commons.math3.util.FastMath;

import java.util.function.BiFunction;

public enum BroadcastableDoubleOperations implements BiFunction<Double, Double, Double> {

    ADD {
        @Override
        public Double apply(Double left, Double right) {
            return left + right;
        }
    },

    SUB {
        @Override
        public Double apply(Double left, Double right) {
            return left - right;
        }
    },

    RSUB {
        @Override
        public Double apply(Double left, Double right) {
            return right - left;
        }
    },

    MUL {
        @Override
        public Double apply(Double left, Double right) {
            return left * right;
        }
    },

    DIV {
        @Override
        public Double apply(Double left, Double right) {
            return left / right;
        }
    },

    RDIV {
        @Override
        public Double apply(Double left, Double right) {
            return right / left;
        }
    },

    LOG_ADD_EXP {
        @Override
        public Double apply(Double a, Double b) {
            double max = Math.max(a, b);
            return max + FastMath.log(FastMath.exp(a - max) + FastMath.exp(b - max));
        }
    },

    LOG_ADD_EXP2 {

        private final double LOG2 = FastMath.log(2);

        @Override
        public Double apply(Double a, Double b) {
            double max = Math.max(a, b);
            return max + FastMath.log(FastMath.pow(2, a - max) + FastMath.pow(2, b - max)) / LOG2;
        }
    },

    GT_MASK {
        @Override
        public Double apply(Double left, Double right) {
            return left > right ? 1.0 : 0.0;
        }
    },

    GTE_MASK {
        @Override
        public Double apply(Double left, Double right) {
            return left >= right ? 1.0 : 0.0;
        }
    },

    LT_MASK {
        @Override
        public Double apply(Double left, Double right) {
            return left < right ? 1.0 : 0.0;
        }
    },

    LTE_MASK {
        @Override
        public Double apply(Double left, Double right) {
            return left <= right ? 1.0 : 0.0;
        }
    }

}
