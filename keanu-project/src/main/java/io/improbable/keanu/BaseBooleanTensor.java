package io.improbable.keanu;

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
    
    BOOLEAN allTrue();

    BOOLEAN allFalse();

    BOOLEAN anyTrue();

    BOOLEAN anyFalse();

    DOUBLE toDoubleMask();

    INTEGER toIntegerMask();
}
