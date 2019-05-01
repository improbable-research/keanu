package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;

import java.util.ArrayList;
import java.util.List;

class TensorMulByMatrixMul {

    static DoubleTensor tensorMmul(DoubleTensor left, DoubleTensor right, int[] dimsLeft, int[] dimsRight) {

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

        DoubleTensor leftTensorAsMatrix = left.permute(leftDimsPermuted).reshape(leftTensorAsMatrixShape);
        DoubleTensor rightTensorAsMatrix = right.permute(rightDimsPermuted).reshape(rightTensorAsMatrixShape);
        DoubleTensor resultAsMatrix = leftTensorAsMatrix.matrixMultiply(rightTensorAsMatrix);

        long[] leftKeptShape = getKeptShape(leftShape, leftDimsKept);
        long[] rightKeptShape = getKeptShape(rightShape, rightDimsKept);

        long[] resultShape = Longs.concat(leftKeptShape, rightKeptShape);
        return resultAsMatrix.reshape(resultShape);
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
