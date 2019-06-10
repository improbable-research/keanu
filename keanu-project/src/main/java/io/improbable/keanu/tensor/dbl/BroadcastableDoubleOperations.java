package io.improbable.keanu.tensor.dbl;

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
