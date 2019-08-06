package io.improbable.keanu.tensor.dbl;

import org.apache.commons.math3.util.FastMath;

import java.util.function.BiFunction;

public enum BroadcastableDoubleOperations implements BiFunction<Double, Double, Double> {


    SAFE_LOG_TIMES {
        @Override
        public Double apply(Double left, Double right) {
            if (right == 0.0) {
                return 0.0;
            } else {
                return FastMath.log(left) * right;
            }
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
    }

}
