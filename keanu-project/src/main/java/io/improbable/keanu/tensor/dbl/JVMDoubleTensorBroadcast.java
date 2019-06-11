package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.TensorShape;
import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMDoubleTensorBroadcast {

    @AllArgsConstructor
    public static class ResultWrapper {
        public final Object outputBuffer;
        public final long[] outputShape;
        public final long[] outputStride;
    }

    public static ResultWrapper broadcastIfNeeded(Object leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                                           Object rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                                           BiFunction op,
                                           boolean inPlace) {
        final boolean needsBroadcast = !Arrays.equals(leftShape, rightShape);

        Object outputBuffer;
        long[] outputShape;
        long[] outputStride;

        if (needsBroadcast) {

            //Short circuit for broadcast with scalars
            if (leftShape.length == 0) {

                outputBuffer = arrayLikeWithLength(rightBuffer, rightBufferLength);
                outputShape = Arrays.copyOf(rightShape, rightShape.length);
                outputStride = Arrays.copyOf(rightStride, rightShape.length);
                scalarLeftAllTypes(leftBuffer, rightBuffer, outputBuffer, op);

            } else if (rightShape.length == 0) {

                outputBuffer = inPlace ? leftBuffer : arrayLikeWithLength(leftBuffer, leftBufferLength);
                outputShape = Arrays.copyOf(leftShape, leftShape.length);
                outputStride = Arrays.copyOf(leftStride, leftStride.length);
                scalarRightAllTypes(leftBuffer, rightBuffer, outputBuffer, op);

            } else {

                return broadcastBinaryOp(
                    leftBuffer, leftShape, leftBufferLength,
                    rightBuffer, rightShape, rightBufferLength,
                    op, inPlace
                );
            }

        } else {
            outputBuffer = inPlace ? leftBuffer : arrayLikeWithLength(leftBuffer, leftBufferLength);
            outputShape = Arrays.copyOf(leftShape, leftShape.length);
            outputStride = Arrays.copyOf(leftStride, leftStride.length);

            elementwiseBinaryOpAllTypes(leftBuffer, rightBuffer, op, outputBuffer);
        }

        return new ResultWrapper(outputBuffer, outputShape, outputStride);
    }

    private static Object arrayLikeWithLength(Object likeThis, int length) {
        if (likeThis instanceof double[]) {
            return new double[length];
        } else {
            throw new IllegalArgumentException("Cannot create array like " + likeThis.getClass().getSimpleName());
        }
    }

    private static void scalarLeftAllTypes(Object leftBuffer, Object rightBuffer, Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            scalarLeftDouble(((double[]) leftBuffer)[0], (double[]) rightBuffer, (double[]) outputBuffer, op);
        } else if (leftBuffer instanceof boolean[]) {
            scalarLeftBoolean(((boolean[]) leftBuffer)[0], (boolean[]) rightBuffer, (boolean[]) outputBuffer, op);
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void scalarLeftDouble(double left, double[] rightBuffer, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarLeftBoolean(boolean left, boolean[] rightBuffer, boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarRightAllTypes(Object leftBuffer, Object rightBuffer, Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            scalarRightDouble((double[]) leftBuffer, ((double[]) rightBuffer)[0], (double[]) outputBuffer, op);
        } else if (leftBuffer instanceof boolean[]) {
            scalarRightBoolean((boolean[]) leftBuffer, ((boolean[]) rightBuffer)[0], (boolean[]) outputBuffer, op);
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void scalarRightDouble(double[] leftBuffer, double right, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    private static void scalarRightBoolean(boolean[] leftBuffer, boolean right, boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    private static void elementwiseBinaryOpAllTypes(Object leftBuffer, Object rightBuffer,
                                                    BiFunction op,
                                                    Object outputBuffer) {

        if (leftBuffer instanceof double[]) {
            elementwiseBinaryOpDouble(
                (double[]) leftBuffer, (double[]) rightBuffer,
                op, (double[]) outputBuffer
            );
        } else if (leftBuffer instanceof boolean[]) {
            elementwiseBinaryOpBoolean(
                (boolean[]) leftBuffer, (boolean[]) rightBuffer,
                op, (boolean[]) outputBuffer
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void elementwiseBinaryOpDouble(double[] leftBuffer, double[] rightBuffer,
                                                  BiFunction<Double, Double, Double> op,
                                                  double[] outputBuffer) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }
    }

    private static void elementwiseBinaryOpBoolean(boolean[] leftBuffer, boolean[] rightBuffer,
                                                   BiFunction<Boolean, Boolean, Boolean> op,
                                                   boolean[] outputBuffer) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }
    }

    private static ResultWrapper broadcastBinaryOp(Object leftBuffer, long[] leftShape, int leftBufferLength,
                                                   Object rightBuffer, long[] rightShape, int rightBufferLength,
                                                   BiFunction op,
                                                   boolean inPlace) {

        //implicitly pad lower ranks with 1s. E.g. [3, 3] & [3] -> [3, 3] -> [1, 3]
        final int resultRank = Math.max(leftShape.length, rightShape.length);
        final long[] paddedLeftShape = TensorShape.shapeToDesiredRankByPrependingOnes(leftShape, resultRank);
        final long[] paddedLeftStride = TensorShape.getRowFirstStride(paddedLeftShape);

        final long[] paddedRightShape = TensorShape.shapeToDesiredRankByPrependingOnes(rightShape, resultRank);
        final long[] paddedRightStride = TensorShape.getRowFirstStride(paddedRightShape);

        final long[] resultShape = Shape.broadcastOutputShape(paddedLeftShape, paddedRightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, paddedLeftShape);

        final Object outputBuffer;
        final long[] outputStride;

        if (resultShapeIsLeftSideShape) {

            outputBuffer = inPlace ? leftBuffer : arrayLikeWithLength(leftBuffer, leftBufferLength);
            outputStride = paddedLeftStride;

            //e.g. [2, 2] * [1, 2]
            broadcastFromRightAllTypes(
                leftBuffer, paddedLeftStride, rightBuffer,
                paddedRightShape, paddedRightStride,
                outputBuffer, op
            );

        } else {

            final boolean resultShapeIsRightSideShape = Arrays.equals(resultShape, paddedRightShape);

            if (resultShapeIsRightSideShape) {

                outputBuffer = arrayLikeWithLength(rightBuffer, rightBufferLength);
                outputStride = paddedRightStride;

                //e.g. [2] / [2, 2]
                broadcastFromLeftAllTypes(
                    leftBuffer, paddedLeftShape, paddedLeftStride,
                    rightBuffer, paddedRightStride,
                    outputBuffer, op
                );

            } else {

                outputBuffer = arrayLikeWithLength(leftBuffer, getLengthAsInt(resultShape));
                outputStride = getRowFirstStride(resultShape);

                //e.g. [2, 2, 1] * [1, 2, 2]
                broadcastFromLeftAndRightAllTypes(
                    leftBuffer, paddedLeftShape, paddedLeftStride,
                    rightBuffer, paddedRightShape, paddedRightStride,
                    outputBuffer, outputStride, op
                );
            }
        }

        return new ResultWrapper(outputBuffer, resultShape, outputStride);
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
    private static void broadcastFromRightAllTypes(Object leftBuffer, long[] leftStride,
                                                   Object rightBuffer, long[] rightShape, long[] rightStride,
                                                   Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            broadcastFromRightDouble(
                (double[]) leftBuffer, leftStride,
                (double[]) rightBuffer, rightShape, rightStride,
                (double[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromRightBoolean(
                (boolean[]) leftBuffer, leftStride,
                (boolean[]) rightBuffer, rightShape, rightStride,
                (boolean[]) outputBuffer, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }


    private static void broadcastFromRightDouble(double[] leftBuffer, long[] leftStride,
                                                 double[] rightBuffer, long[] rightShape, long[] rightStride,
                                                 double[] outputBuffer, BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }
    }

    private static void broadcastFromRightBoolean(boolean[] leftBuffer, long[] leftStride,
                                                  boolean[] rightBuffer, long[] rightShape, long[] rightStride,
                                                  boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);

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
    private static void broadcastFromLeftAllTypes(Object leftBuffer, long[] leftShape, long[] leftStride,
                                                  Object rightBuffer, long[] rightStride,
                                                  Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            broadcastFromLeftDouble(
                (double[]) leftBuffer, leftShape, leftStride,
                (double[]) rightBuffer, rightStride,
                (double[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromLeftBoolean(
                (boolean[]) leftBuffer, leftShape, leftStride,
                (boolean[]) rightBuffer, rightStride,
                (boolean[]) outputBuffer, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }


    private static void broadcastFromLeftDouble(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                                double[] rightBuffer, long[] rightStride,
                                                double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }
    }

    private static void broadcastFromLeftBoolean(boolean[] leftBuffer, long[] leftShape, long[] leftStride,
                                                 boolean[] rightBuffer, long[] rightStride,
                                                 boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

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
    private static void broadcastFromLeftAndRightAllTypes(Object leftBuffer, long[] leftShape, long[] leftStride,
                                                          Object rightBuffer, long[] rightShape, long[] rightStride,
                                                          Object outputBuffer, long[] outputStride,
                                                          BiFunction op) {

        if (leftBuffer instanceof double[]) {
            broadcastFromLeftAndRightDouble(
                (double[]) leftBuffer, leftShape, leftStride,
                (double[]) rightBuffer, rightShape, rightStride,
                (double[]) outputBuffer, outputStride, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromLeftAndRightBoolean(
                (boolean[]) leftBuffer, leftShape, leftStride,
                (boolean[]) rightBuffer, rightShape, rightStride,
                (boolean[]) outputBuffer, outputStride, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void broadcastFromLeftAndRightDouble(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                                        double[] rightBuffer, long[] rightShape, long[] rightStride,
                                                        double[] outputBuffer, long[] outputStride,
                                                        BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

    private static void broadcastFromLeftAndRightBoolean(boolean[] leftBuffer, long[] leftShape, long[] leftStride,
                                                         boolean[] rightBuffer, long[] rightShape, long[] rightStride,
                                                         boolean[] outputBuffer, long[] outputStride,
                                                         BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

}
