package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public enum Operation {
    PLUS {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.plus(rhs);
        }
        public double apply(double lhs, double rhs) {
            return lhs + rhs;
        }
    },
    MINUS {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.minus(rhs);
        }
        public double apply(double lhs, double rhs) {
            return lhs - rhs;
        }
    },
    TIMES {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.times(rhs);
        }
        public double apply(double lhs, double rhs) {
            return lhs * rhs;
        }
    },
    DIVIDE {
        public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
            return lhs.div(rhs);
        }
        public double apply(double lhs, double rhs) {
            return lhs / rhs;
        }
    };

    public abstract DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs);
    public abstract double apply(double lhs, double rhs);
}
