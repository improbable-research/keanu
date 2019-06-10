package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.TensorShape;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMDoubleTensorBroadcast {

    static JVMDoubleTensor broadcastScalar(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                           double[] rightBuffer, long[] rightShape, long[] rightStride,
                                           BiFunction<Double, Double, Double> op,
                                           boolean inPlace) {
        final double[] outputBuffer;
        final long[] resultShape;
        final long[] resultStride;

        if (leftShape.length == 0) {
            outputBuffer = new double[rightBuffer.length];
            resultShape = Arrays.copyOf(rightShape, rightShape.length);
            resultStride = Arrays.copyOf(rightStride, rightShape.length);
            scalarLeft(leftBuffer[0], rightBuffer, outputBuffer, op);
        } else {
            outputBuffer = inPlace ? leftBuffer : new double[leftBuffer.length];
            resultShape = Arrays.copyOf(leftShape, leftShape.length);
            resultStride = leftStride;
            scalarRight(leftBuffer, rightBuffer[0], outputBuffer, op);
        }

        return new JVMDoubleTensor(outputBuffer, resultShape, resultStride);
    }

    private static void scalarLeft(double left, double[] rightBuffer, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarRight(double[] leftBuffer, double right, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    static JVMDoubleTensor elementwiseBinaryOp(double[] leftBuffer, double[] rightBuffer, long[] shape, long[] stride,
                                               BiFunction<Double, Double, Double> op,
                                               boolean inPlace) {

        final double[] outputBuffer = inPlace ? leftBuffer : new double[leftBuffer.length];

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }

        return new JVMDoubleTensor(outputBuffer, shape, stride);
    }

    static JVMDoubleTensor broadcastBinaryOp(double[] leftBuffer, long[] leftShape,
                                             double[] rightBuffer, long[] rightShape,
                                             BiFunction<Double, Double, Double> op,
                                             boolean inPlace) {

        //implicitly pad lower ranks with 1s. E.g. [3, 3] & [3] -> [3, 3] -> [1, 3]
        final int resultRank = Math.max(leftShape.length, rightShape.length);
        final long[] paddedLeftShape = getShapeOrPadToRank(leftShape, resultRank);
        final long[] paddedLeftStride = TensorShape.getRowFirstStride(paddedLeftShape);

        final long[] paddedRightShape = getShapeOrPadToRank(rightShape, resultRank);
        final long[] paddedRightStride = TensorShape.getRowFirstStride(paddedRightShape);

        final long[] resultShape = Shape.broadcastOutputShape(paddedLeftShape, paddedRightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, paddedLeftShape);

        final double[] outputBuffer;
        final long[] outputStride;

        if (resultShapeIsLeftSideShape) {

            outputBuffer = inPlace ? leftBuffer : new double[leftBuffer.length];
            outputStride = paddedLeftStride;

            //e.g. [2, 2] * [1, 2]
            broadcastFromRight(
                leftBuffer, paddedLeftStride, rightBuffer,
                paddedRightShape, paddedRightStride,
                outputBuffer, op
            );

        } else {

            final boolean resultShapeIsRightSideShape = Arrays.equals(resultShape, paddedRightShape);
            outputBuffer = new double[getLengthAsInt(resultShape)];

            if (resultShapeIsRightSideShape) {

                outputStride = paddedRightStride;

                //e.g. [2] / [2, 2]
                broadcastFromLeft(
                    leftBuffer, paddedLeftShape, paddedLeftStride,
                    rightBuffer, paddedRightStride,
                    outputBuffer, op
                );

            } else {

                outputStride = getRowFirstStride(resultShape);

                //e.g. [2, 2, 1] * [1, 2, 2]
                broadcastFromLeftAndRight(
                    leftBuffer, paddedLeftShape, paddedLeftStride,
                    rightBuffer, paddedRightShape, paddedRightStride,
                    outputBuffer, outputStride, op
                );
            }
        }

        return new JVMDoubleTensor(outputBuffer, resultShape, outputStride);
    }

    private static long[] getShapeOrPadToRank(long[] shape, int rank) {
        if (shape.length == rank) {
            return shape;
        } else {
            return TensorShape.shapeToDesiredRankByPrependingOnes(shape, rank);
        }
    }

    /**
     * The broadcast result shape is equal to the left operand shape.
     * <p>
     * e.g. [2, 2] * [1, 2]
     *
     * @param leftBuffer
     * @param leftStride
     * @param rightBuffer
     * @param rightShape
     * @param rightStride
     * @param outputBuffer
     * @param op
     * @return
     */
    private static void broadcastFromRight(double[] leftBuffer, long[] leftStride,
                                           double[] rightBuffer, long[] rightShape, long[] rightStride,
                                           double[] outputBuffer, BiFunction<Double, Double, Double> op) {
        Preconditions.checkArgument(leftBuffer.length >= rightBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = mapBroadcastIndex(i, leftStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }
    }

    /**
     * The broadcast result shape is equal to the right operand shape.
     * <p>
     * e.g. [2] / [2, 2]
     *
     * @param leftBuffer
     * @param leftShape
     * @param leftStride
     * @param rightBuffer
     * @param rightStride
     * @param outputBuffer
     * @param op
     * @return
     */
    private static void broadcastFromLeft(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                          double[] rightBuffer, long[] rightStride,
                                          double[] outputBuffer, BiFunction<Double, Double, Double> op) {
        Preconditions.checkArgument(leftBuffer.length <= rightBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = mapBroadcastIndex(i, rightStride, leftShape, leftStride);

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }
    }

    /**
     * Neither the left operand shape nor the right operand shape equal the result shape.
     * <p>
     * e.g. [2, 2, 1] * [2, 2] = [2, 2, 2]
     *
     * @param leftBuffer
     * @param leftShape
     * @param leftStride
     * @param rightBuffer
     * @param rightShape
     * @param rightStride
     * @param outputBuffer
     * @param outputStride
     * @param op
     */
    private static void broadcastFromLeftAndRight(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                                  double[] rightBuffer, long[] rightShape, long[] rightStride,
                                                  double[] outputBuffer, long[] outputStride,
                                                  BiFunction<Double, Double, Double> op) {

        Preconditions.checkArgument(leftBuffer.length <= outputBuffer.length);
        Preconditions.checkArgument(rightBuffer.length <= outputBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {

            int k = mapBroadcastIndex(i, outputStride, leftShape, leftStride);
            int j = mapBroadcastIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

    private static int mapBroadcastIndex(int fromFlatIndex, long[] fromStride, long[] toShape, long[] toStride) {

        final long[] fromShapeIndex = new long[fromStride.length];
        final long[] toShapeIndex = new long[fromShapeIndex.length];
        int remainder = fromFlatIndex;
        int toFlatIndex = 0;

        for (int i = 0; i < fromStride.length; i++) {
            fromShapeIndex[i] = remainder / fromStride[i];
            remainder -= fromShapeIndex[i] * fromStride[i];
            toShapeIndex[i] = fromShapeIndex[i] % toShape[i];
            toFlatIndex += toStride[i] * toShapeIndex[i];
        }

        return toFlatIndex;
    }

}
