package io.improbable.keanu.tensor.bool;

import java.util.function.BiFunction;

public enum BroadcastableBooleanOperations implements BiFunction<Boolean, Boolean, Boolean> {

    AND {
        @Override
        public Boolean apply(Boolean left, Boolean right) {
            return left && right;
        }
    },

    OR {
        @Override
        public Boolean apply(Boolean left, Boolean right) {
            return left || right;
        }
    },

    XOR {
        @Override
        public Boolean apply(Boolean left, Boolean right) {
            return left ^ right;
        }
    },
}
