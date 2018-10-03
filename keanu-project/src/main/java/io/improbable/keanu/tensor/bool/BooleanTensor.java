package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface BooleanTensor extends Tensor<Boolean> {

  static BooleanTensor create(boolean value, int[] shape) {
    return new SimpleBooleanTensor(value, shape);
  }

  static BooleanTensor create(boolean[] values, int... shape) {
    return new SimpleBooleanTensor(values, shape);
  }

  static BooleanTensor create(boolean[] values) {
    return create(values, 1, values.length);
  }

  static BooleanTensor scalar(boolean scalarValue) {
    return new SimpleBooleanTensor(scalarValue);
  }

  static BooleanTensor placeHolder(int[] shape) {
    return new SimpleBooleanTensor(shape);
  }

  static BooleanTensor trues(int... shape) {
    return new SimpleBooleanTensor(true, shape);
  }

  static BooleanTensor falses(int... shape) {
    return new SimpleBooleanTensor(false, shape);
  }

  static BooleanTensor concat(int dimension, BooleanTensor[] toConcat) {
    DoubleTensor[] toDoubles = new DoubleTensor[toConcat.length];

    for (int i = 0; i < toConcat.length; i++) {
      toDoubles[i] = toConcat[i].toDoubleMask();
    }

    DoubleTensor concat = DoubleTensor.concat(dimension, toDoubles);
    double[] concatFlat = concat.asFlatDoubleArray();
    boolean[] data = new boolean[concat.asFlatDoubleArray().length];

    for (int i = 0; i < data.length; i++) {
      data[i] = concatFlat[i] == 1.0;
    }

    return new SimpleBooleanTensor(data, concat.getShape());
  }

  @Override
  BooleanTensor reshape(int... newShape);

  @Override
  BooleanTensor duplicate();

  BooleanTensor and(BooleanTensor that);

  BooleanTensor or(BooleanTensor that);

  BooleanTensor not();

  DoubleTensor setDoubleIf(DoubleTensor trueValue, DoubleTensor falseValue);

  IntegerTensor setIntegerIf(IntegerTensor trueValue, IntegerTensor falseValue);

  BooleanTensor setBooleanIf(BooleanTensor trueValue, BooleanTensor falseValue);

  <T> Tensor<T> setIf(Tensor<T> trueValue, Tensor<T> falseValue);

  BooleanTensor andInPlace(BooleanTensor that);

  BooleanTensor orInPlace(BooleanTensor that);

  BooleanTensor notInPlace();

  boolean allTrue();

  boolean allFalse();

  DoubleTensor toDoubleMask();

  IntegerTensor toIntegerMask();

  @Override
  BooleanTensor slice(int dimension, int index);
}
