package io.improbable.keanu.tensor.lng;

import java.util.function.BiFunction;

public enum BroadcastableLongOperations implements BiFunction<Long, Long, Long> {

    ADD {
        @Override
        public Long apply(Long left, Long right) {
            return left + right;
        }
    },

    SUB {
        @Override
        public Long apply(Long left, Long right) {
            return left - right;
        }
    },

    RSUB {
        @Override
        public Long apply(Long left, Long right) {
            return right - left;
        }
    },

    MUL {
        @Override
        public Long apply(Long left, Long right) {
            return left * right;
        }
    },

    DIV {
        @Override
        public Long apply(Long left, Long right) {
            return left / right;
        }
    },

    RDIV {
        @Override
        public Long apply(Long left, Long right) {
            return right / left;
        }
    },

    POW {
        @Override
        public Long apply(Long left, Long right) {
            if (right == 0) {
                return 1L;
            }

            long result = left;
            for (long i = 0; i < right - 1; i++) {
                result *= left;
            }

            return result;
        }

    },

    GT_MASK {
        @Override
        public Long apply(Long left, Long right) {
            return left > right ? 1L : 0L;
        }
    },

    GTE_MASK {
        @Override
        public Long apply(Long left, Long right) {
            return left >= right ? 1L : 0L;
        }
    },

    LT_MASK {
        @Override
        public Long apply(Long left, Long right) {
            return left < right ? 1L : 0L;
        }
    },

    LTE_MASK {
        @Override
        public Long apply(Long left, Long right) {
            return left <= right ? 1L : 0L;
        }
    }

}
