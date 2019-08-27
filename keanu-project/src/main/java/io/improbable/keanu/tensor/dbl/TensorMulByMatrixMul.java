package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.tensor.NumberTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorMulByMatrixMul {

    public static <T extends Number, TENSOR extends NumberTensor<T, TENSOR>> TENSOR tensorMmul(TENSOR left, TENSOR right, int[] dimsLeft, int[] dimsRight) {

        long[] leftShape = left.getShape();
        long[] rightShape = right.getShape();

        validateTensorMmul(leftShape, rightShape, dimsLeft, dimsRight);

        List<Integer> leftDimsKept = getKeptDimensions(leftShape.length, dimsLeft);
        List<Integer> rightDimsKept = getKeptDimensions(rightShape.length, dimsRight);

        int[] leftDimsPermuted = Ints.concat(Ints.toArray(leftDimsKept), dimsLeft);
        int[] rightDimsPermuted = Ints.concat(dimsRight, Ints.toArray(rightDimsKept));

        long dimsLength = calculateDimensionsLength(leftShape, dimsLeft);
        long[] leftTensorAsMatrixShape = {-1, dimsLength};
        long[] rightTensorAsMatrixShape = {dimsLength, -1};

        TENSOR leftTensorAsMatrix = left.permute(leftDimsPermuted).reshape(leftTensorAsMatrixShape);
        TENSOR rightTensorAsMatrix = right.permute(rightDimsPermuted).reshape(rightTensorAsMatrixShape);
        TENSOR resultAsMatrix = leftTensorAsMatrix.matrixMultiply(rightTensorAsMatrix);

        long[] leftKeptShape = getKeptShape(leftShape, leftDimsKept);
        long[] rightKeptShape = getKeptShape(rightShape, rightDimsKept);

        long[] resultShape = Longs.concat(leftKeptShape, rightKeptShape);
        return resultAsMatrix.reshape(resultShape);
    }

    public static long[] getResultShape(long[] leftShape, long[] rightShape, int[] dimsLeft, int[] dimsRight) {

        List<Integer> leftDimsKept = getKeptDimensions(leftShape.length, dimsLeft);
        List<Integer> rightDimsKept = getKeptDimensions(rightShape.length, dimsRight);

        long[] leftKeptShape = getKeptShape(leftShape, leftDimsKept);
        long[] rightKeptShape = getKeptShape(rightShape, rightDimsKept);

        return Longs.concat(leftKeptShape, rightKeptShape);
    }

    /**
     * Validates that the multiply dimensions match for the left shape and the right shape. Also, corrects for the
     * use of a negative dimension to represent dimensions from the right.
     *
     * @param leftShape  the shape of the left tensor being mmul
     * @param rightShape the shape of the right tensor being mmul
     * @param dimsLeft   dimensions along the left to multiply
     * @param dimsRight  dimensions along the right to multiply
     */
    private static void validateTensorMmul(long[] leftShape, long[] rightShape, int[] dimsLeft, int[] dimsRight) {

        int validationLength = Math.min(dimsLeft.length, dimsRight.length);
        for (int i = 0; i < validationLength; i++) {

            if (leftShape.length <= dimsLeft[i]) {
                throw new IllegalArgumentException(
                    "Invalid left dimension " + dimsLeft[i] + " for shape " + Arrays.toString(leftShape)
                );
            }

            if (rightShape.length <= dimsRight[i]) {
                throw new IllegalArgumentException(
                    "Invalid right dimension " + dimsRight[i] + " for shape " + Arrays.toString(rightShape)
                );
            }

            if (leftShape[dimsLeft[i]] != rightShape[dimsRight[i]]) {
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            }

            if (dimsLeft[i] < 0) {
                dimsLeft[i] += leftShape.length;
            }

            if (dimsRight[i] < 0) {
                dimsRight[i] += rightShape.length;
            }

        }
    }

    private static List<Integer> getKeptDimensions(int shapeLength, int[] dims) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < shapeLength; i++) {
            if (!Ints.contains(dims, i)) {
                result.add(i);
            }
        }
        return result;
    }

    private static long[] getKeptShape(long[] shape, List<Integer> keptDimensions) {
        long[] keptShape = Longs.toArray(keptDimensions);
        for (int i = 0; i < keptShape.length; i++) {
            keptShape[i] = shape[Ints.checkedCast(keptShape[i])];
        }
        return keptShape;
    }

    private static long calculateDimensionsLength(long[] shape, int[] dims) {
        long length = 1;
        int aLength = Math.min(shape.length, dims.length);
        for (int i = 0; i < aLength; i++) {
            length *= shape[dims[i]];
        }
        return length;
    }

}
