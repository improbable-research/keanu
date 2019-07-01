package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

public abstract class Nd4jNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>> extends Nd4jTensor<T, TENSOR> implements NumberTensor<T, TENSOR> {
    public Nd4jNumberTensor(INDArray tensor) {
        super(tensor);
    }

    @Override
    public TENSOR setWithMaskInPlace(TENSOR mask, T value) {
        if (this.getLength() != mask.getLength()) {
            throw new IllegalArgumentException("The lengths of the tensor and mask must match, but got tensor length: " + this.getLength() + ", mask length: " + mask.getLength());
        }

        INDArray maskINDArray = getTensor(mask);

        //Nd4j compare and set only works for fp types
        INDArray dblBuffer = tensor.dataType() == DataType.DOUBLE ? tensor : tensor.castTo(DataType.DOUBLE);
        INDArray dblMask = maskINDArray.dataType() == DataType.DOUBLE ? maskINDArray : maskINDArray.castTo(DataType.DOUBLE);
        double dblValue = value.doubleValue();

        double trueValue = 1.0;
        if (dblValue == 0.0) {
            trueValue = 1.0 - trueValue;
            dblMask.negi().addi(1.0);
        }
        double falseValue = 1.0 - trueValue;

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblMask, dblValue, Conditions.equals(trueValue))
        );

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblBuffer, dblMask, Conditions.notEquals(falseValue))
        );

        return set(dblBuffer);
    }
}
