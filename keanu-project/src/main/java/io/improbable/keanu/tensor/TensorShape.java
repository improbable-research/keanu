package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorShape {

  private int[] shape;

  public TensorShape(int[] shape) {
    this.shape = Arrays.copyOf(shape, shape.length);
  }

  public int[] getShape() {
    return Arrays.copyOf(shape, shape.length);
  }

  public boolean isScalar() {
    return isScalar(shape);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    TensorShape that = (TensorShape) o;

    return Arrays.equals(shape, that.shape);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(shape);
  }

  /**
   * @param shape for finding length of
   * @return the number of elements in a tensor. This is the product of all ints in shape.
   */
  public static long getLength(int[] shape) {
    if (shape.length == 0) {
      return 0;
    } else {
      long length = 1;
      for (int dim : shape) {
        length *= dim;
      }
      return length;
    }
  }

  /**
   * @param shape shape to find stride for
   * @return the stride which is used to convert from a N dimensional index to a buffer array flat
   *     index. This is based on the C convention of row first instead of column.
   */
  public static int[] getRowFirstStride(int[] shape) {
    int[] stride = new int[shape.length];
    stride[stride.length - 1] = 1;

    int buffer = 1;
    for (int i = stride.length - 2; i >= 0; i--) {
      buffer *= shape[i + 1];
      stride[i] = buffer;
    }

    return stride;
  }

  /**
   * @param shape shape to find the index for
   * @param stride stride to find the index for
   * @param index the index in each dimension
   * @return the flat index from a N dimensional index
   */
  public static int getFlatIndex(int[] shape, int[] stride, int... index) {
    int flatIndex = 0;
    for (int i = 0; i < index.length; i++) {

      if (index[i] >= shape[i]) {
        throw new IllegalArgumentException(
            "Invalid index " + Arrays.toString(index) + " for shape " + Arrays.toString(shape));
      }

      flatIndex += stride[i] * index[i];
    }
    return flatIndex;
  }

  /**
   * This method can be interpreted as the opposite to getFlatIndex.
   *
   * @param shape the shape to find the index of
   * @param stride the stride to find the index of
   * @param flatIndex the index to f
   * @return converts from a flat index to a N dimensional index. Where N = the dimensionality of
   *     the shape.
   */
  public static int[] getShapeIndices(int[] shape, int[] stride, int flatIndex) {
    if (flatIndex > getLength(shape)) {
      throw new IllegalArgumentException("The requested index is out of the bounds of this shape.");
    }
    int[] shapeIndices = new int[stride.length];
    int remainder = flatIndex;
    for (int i = 0; i < stride.length; i++) {
      shapeIndices[i] = remainder / stride[i];
      remainder -= shapeIndices[i] * stride[i];
    }
    return shapeIndices;
  }

  public static boolean isScalar(int[] shape) {
    return getLength(shape) == 1;
  }

  public static int[] concat(int[] shape1, int[] shape2) {
    int[] result = new int[shape1.length + shape2.length];
    System.arraycopy(shape1, 0, result, 0, shape1.length);
    System.arraycopy(shape2, 0, result, shape1.length, shape2.length);
    return result;
  }

  /**
   * @param fromDimension starting from and including this dimension
   * @param toDimension up to but excluding this dimension
   * @return an int array containing the dimension numbers from a given dimension to a higher
   *     dimension. e.g. dimensionRange(0, 3) = int[]{0, 1, 2}
   */
  public static int[] dimensionRange(int fromDimension, int toDimension) {
    if (fromDimension > toDimension) {
      throw new IllegalArgumentException("from dimension must be less than to dimension");
    }

    int dimensionCount = toDimension - fromDimension;
    int[] dims = new int[dimensionCount];
    for (int i = 0; i < dimensionCount; i++) {
      dims[i] = i + fromDimension;
    }
    return dims;
  }

  public static int[] selectDimensions(int from, int to, int[] shape) {
    if (from > to) {
      throw new IllegalArgumentException("to dimension must be less than from");
    }

    int[] newShape = new int[to - from + 1];

    for (int i = 0; i < (to - from + 1); i++) {
      newShape[i] = shape[i + from];
    }

    return newShape;
  }

  public static int[] slideDimension(int from, int to, int[] shape) {
    List<Integer> shapeList = new ArrayList<>(Ints.asList(shape));
    Integer dimLength = shapeList.remove(from);
    shapeList.add(to, dimLength);
    return Ints.toArray(shapeList);
  }

  public static int[] shapeDesiredToRankByAppendingOnes(int[] lowRankTensorShape, int desiredRank) {
    return increaseRankByPaddingOnes(lowRankTensorShape, desiredRank, true);
  }

  public static int[] shapeToDesiredRankByPrependingOnes(
      int[] lowRankTensorShape, int desiredRank) {
    return increaseRankByPaddingOnes(lowRankTensorShape, desiredRank, false);
  }

  private static int[] increaseRankByPaddingOnes(
      int[] lowRankTensorShape, int desiredRank, boolean append) {
    int[] paddedShape = new int[desiredRank];
    if (lowRankTensorShape.length > desiredRank) {
      throw new IllegalArgumentException(
          "low rank tensor must be rank less than or equal to desired rank");
    }

    Arrays.fill(paddedShape, 1);
    if (append) {
      System.arraycopy(lowRankTensorShape, 0, paddedShape, 0, lowRankTensorShape.length);
    } else {
      System.arraycopy(
          lowRankTensorShape,
          0,
          paddedShape,
          paddedShape.length - lowRankTensorShape.length,
          lowRankTensorShape.length);
    }

    return paddedShape;
  }

  public static int[] shapeSlice(int dimension, int[] shape) {
    int[] newShape = Arrays.copyOf(shape, shape.length);
    newShape[dimension] = 1;
    return newShape;
  }

  /**
   * Writes a lower rank shape over a higher rank shape, starting from the right.
   *
   * <p>e.g: high rank shape = [1, 2, 2, 1] low rank shape = [1, 4]
   *
   * <p>Result after copy = [1, 2, 1, 4]
   *
   * @param higherRankShape source shape that will get written over
   * @param lowerRankShape shape to write
   * @return the high rank shape with the lower rank shape inserted on top of it
   */
  public static int[] copyLowRankOverHighRankFromTailEnd(
      int[] higherRankShape, int[] lowerRankShape) {
    int[] highRankCopy = Arrays.copyOf(higherRankShape, higherRankShape.length);
    int deltaLength = highRankCopy.length - lowerRankShape.length;
    System.arraycopy(
        lowerRankShape, 0, highRankCopy, deltaLength, highRankCopy.length - deltaLength);
    return highRankCopy;
  }
}
