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
    
    BOOLEAN allTrue();

    BOOLEAN allFalse();

    BOOLEAN anyTrue();

    BOOLEAN anyFalse();

    DOUBLE toDoubleMask();

    INTEGER toIntegerMask();
}
