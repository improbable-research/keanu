package io.improbable.keanu;

import io.improbable.keanu.tensor.Tensor;

public interface BaseBooleanTensor<
    BOOLEAN extends BaseTensor<BOOLEAN, Boolean, BOOLEAN>,
    INTEGER extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Integer, INTEGER>,
    DOUBLE extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Double, DOUBLE>
    > extends BaseTensor<BOOLEAN, Boolean, BOOLEAN> {

    BOOLEAN and(BOOLEAN that);

    BOOLEAN and(boolean that);

    BOOLEAN or(BOOLEAN that);

    BOOLEAN or(boolean that);

    BOOLEAN xor(BOOLEAN that);

    BOOLEAN not();

    DOUBLE doubleWhere(DOUBLE trueValue, DOUBLE falseValue);

    INTEGER integerWhere(INTEGER trueValue, INTEGER falseValue);

    BOOLEAN booleanWhere(BOOLEAN trueValue, BOOLEAN falseValue);

    <T, TENSOR extends Tensor<T, TENSOR>> TENSOR where(TENSOR trueValue, TENSOR falseValue);

    BOOLEAN allTrue();

    BOOLEAN allFalse();

    BOOLEAN anyTrue();

    BOOLEAN anyFalse();

    DOUBLE toDoubleMask();

    INTEGER toIntegerMask();
}
