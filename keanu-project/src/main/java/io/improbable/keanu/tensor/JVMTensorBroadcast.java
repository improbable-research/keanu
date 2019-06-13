package io.improbable.keanu.tensor;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMTensorBroadcast {

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
                    leftBuffer, leftShape, leftStride, leftBufferLength,
                    rightBuffer, rightShape, rightStride, rightBufferLength,
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
        } else if (likeThis instanceof boolean[]) {
            return new boolean[length];
        } else if (likeThis instanceof Object[]) {
            return new Object[length];
        } else {
            throw new IllegalArgumentException("Cannot create array like " + likeThis.getClass().getSimpleName());
        }
    }

    private static void scalarLeftAllTypes(Object leftBuffer, Object rightBuffer, Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            scalarLeft(((double[]) leftBuffer)[0], (double[]) rightBuffer, (double[]) outputBuffer, op);
        } else if (leftBuffer instanceof boolean[]) {
            scalarLeft(((boolean[]) leftBuffer)[0], (boolean[]) rightBuffer, (boolean[]) outputBuffer, op);
        } else if (leftBuffer instanceof Object[]) {
            scalarLeft(((Object[]) leftBuffer)[0], (Object[]) rightBuffer, (Object[]) outputBuffer, op);
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void scalarLeft(double left, double[] rightBuffer, double[] outputBuffer,
                                   BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarLeft(boolean left, boolean[] rightBuffer, boolean[] outputBuffer,
                                   BiFunction<Boolean, Boolean, Boolean> op) {
        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarLeft(Object left, Object[] rightBuffer, Object[] outputBuffer,
                                   BiFunction op) {
        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }
    }

    private static void scalarRightAllTypes(Object leftBuffer, Object rightBuffer, Object outputBuffer, BiFunction op) {

        if (leftBuffer instanceof double[]) {
            scalarRight((double[]) leftBuffer, ((double[]) rightBuffer)[0], (double[]) outputBuffer, op);
        } else if (leftBuffer instanceof boolean[]) {
            scalarRight((boolean[]) leftBuffer, ((boolean[]) rightBuffer)[0], (boolean[]) outputBuffer, op);
        } else if (leftBuffer instanceof Object[]) {
            scalarRight((Object[]) leftBuffer, ((Object[]) rightBuffer)[0], (Object[]) outputBuffer, op);
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void scalarRight(double[] leftBuffer, double right, double[] outputBuffer,
                                    BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    private static void scalarRight(boolean[] leftBuffer, boolean right, boolean[] outputBuffer,
                                    BiFunction<Boolean, Boolean, Boolean> op) {
        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    private static void scalarRight(Object[] leftBuffer, Object right, Object[] outputBuffer,
                                    BiFunction op) {
        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }
    }

    private static void elementwiseBinaryOpAllTypes(Object leftBuffer, Object rightBuffer,
                                                    BiFunction op,
                                                    Object outputBuffer) {

        if (leftBuffer instanceof double[]) {
            elementwiseBinaryOp(
                (double[]) leftBuffer, (double[]) rightBuffer,
                op, (double[]) outputBuffer
            );
        } else if (leftBuffer instanceof boolean[]) {
            elementwiseBinaryOp(
                (boolean[]) leftBuffer, (boolean[]) rightBuffer,
                op, (boolean[]) outputBuffer
            );
        } else if (leftBuffer instanceof Object[]) {
            elementwiseBinaryOp(
                (Object[]) leftBuffer, (Object[]) rightBuffer,
                op, (Object[]) outputBuffer
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void elementwiseBinaryOp(double[] leftBuffer, double[] rightBuffer,
                                            BiFunction<Double, Double, Double> op,
                                            double[] outputBuffer) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }
    }

    private static void elementwiseBinaryOp(boolean[] leftBuffer, boolean[] rightBuffer,
                                            BiFunction<Boolean, Boolean, Boolean> op,
                                            boolean[] outputBuffer) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }
    }

    private static void elementwiseBinaryOp(Object[] leftBuffer, Object[] rightBuffer,
                                            BiFunction op,
                                            Object[] outputBuffer) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }
    }

    private static ResultWrapper broadcastBinaryOp(Object leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                                                   Object rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                                                   BiFunction op,
                                                   boolean inPlace) {

        final long[] resultShape = Shape.broadcastOutputShape(leftShape, rightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, leftShape);

        final Object outputBuffer;
        final long[] outputStride;

        if (resultShapeIsLeftSideShape) {

            outputBuffer = inPlace ? leftBuffer : arrayLikeWithLength(leftBuffer, leftBufferLength);
            outputStride = leftStride;

            //e.g. [2, 2] * [1, 2]
            broadcastFromRightAllTypes(
                leftBuffer, leftStride, rightBuffer,
                rightShape, rightStride,
                outputBuffer, op
            );

        } else {

            final boolean resultShapeIsRightSideShape = Arrays.equals(resultShape, rightShape);

            if (resultShapeIsRightSideShape) {

                outputBuffer = arrayLikeWithLength(rightBuffer, rightBufferLength);
                outputStride = rightStride;

                //e.g. [2] / [2, 2]
                broadcastFromLeftAllTypes(
                    leftBuffer, leftShape, leftStride,
                    rightBuffer, rightStride,
                    outputBuffer, op
                );

            } else {

                outputBuffer = arrayLikeWithLength(leftBuffer, getLengthAsInt(resultShape));
                outputStride = getRowFirstStride(resultShape);

                //e.g. [2, 2, 1] * [1, 2, 2]
                broadcastFromLeftAndRightAllTypes(
                    leftBuffer, leftShape, leftStride,
                    rightBuffer, rightShape, rightStride,
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
            broadcastFromRight(
                (double[]) leftBuffer, leftStride,
                (double[]) rightBuffer, rightShape, rightStride,
                (double[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromRight(
                (boolean[]) leftBuffer, leftStride,
                (boolean[]) rightBuffer, rightShape, rightStride,
                (boolean[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof Object[]) {
            broadcastFromRight(
                (Object[]) leftBuffer, leftStride,
                (Object[]) rightBuffer, rightShape, rightStride,
                (Object[]) outputBuffer, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }


    private static void broadcastFromRight(double[] leftBuffer, long[] leftStride,
                                           double[] rightBuffer, long[] rightShape, long[] rightStride,
                                           double[] outputBuffer, BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }
    }

    private static void broadcastFromRight(boolean[] leftBuffer, long[] leftStride,
                                           boolean[] rightBuffer, long[] rightShape, long[] rightStride,
                                           boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }
    }

    private static void broadcastFromRight(Object[] leftBuffer, long[] leftStride,
                                           Object[] rightBuffer, long[] rightShape, long[] rightStride,
                                           Object[] outputBuffer, BiFunction op) {
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
            broadcastFromLeft(
                (double[]) leftBuffer, leftShape, leftStride,
                (double[]) rightBuffer, rightStride,
                (double[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromLeft(
                (boolean[]) leftBuffer, leftShape, leftStride,
                (boolean[]) rightBuffer, rightStride,
                (boolean[]) outputBuffer, op
            );
        } else if (leftBuffer instanceof Object[]) {
            broadcastFromLeft(
                (Object[]) leftBuffer, leftShape, leftStride,
                (Object[]) rightBuffer, rightStride,
                (Object[]) outputBuffer, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    public static void broadcast(double[] buffer, long[] shape, long[] stride,
                                 double[] outputBuffer, long[] outputStride) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, outputStride, shape, stride);

            outputBuffer[i] = buffer[j];
        }
    }

    public static void broadcast(boolean[] buffer, long[] shape, long[] stride,
                                 boolean[] outputBuffer, long[] outputStride) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, outputStride, shape, stride);

            outputBuffer[i] = buffer[j];
        }
    }

    public static void broadcast(Object[] buffer, long[] shape, long[] stride,
                                 Object[] outputBuffer, long[] outputShape, long[] outputStride) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, outputStride, shape, stride);

            outputBuffer[i] = buffer[j];
        }
    }

    private static void broadcastFromLeft(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                          double[] rightBuffer, long[] rightStride,
                                          double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }
    }

    private static void broadcastFromLeft(boolean[] leftBuffer, long[] leftShape, long[] leftStride,
                                          boolean[] rightBuffer, long[] rightStride,
                                          boolean[] outputBuffer, BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }
    }

    private static void broadcastFromLeft(Object[] leftBuffer, long[] leftShape, long[] leftStride,
                                          Object[] rightBuffer, long[] rightStride,
                                          Object[] outputBuffer, BiFunction op) {

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
            broadcastFromLeftAndRight(
                (double[]) leftBuffer, leftShape, leftStride,
                (double[]) rightBuffer, rightShape, rightStride,
                (double[]) outputBuffer, outputStride, op
            );
        } else if (leftBuffer instanceof boolean[]) {
            broadcastFromLeftAndRight(
                (boolean[]) leftBuffer, leftShape, leftStride,
                (boolean[]) rightBuffer, rightShape, rightStride,
                (boolean[]) outputBuffer, outputStride, op
            );
        } else if (leftBuffer instanceof Object[]) {
            broadcastFromLeftAndRight(
                (Object[]) leftBuffer, leftShape, leftStride,
                (Object[]) rightBuffer, rightShape, rightStride,
                (Object[]) outputBuffer, outputStride, op
            );
        } else {
            throw new IllegalArgumentException("Cannot broadcast for data type " + leftBuffer.getClass().getSimpleName());
        }
    }

    private static void broadcastFromLeftAndRight(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                                  double[] rightBuffer, long[] rightShape, long[] rightStride,
                                                  double[] outputBuffer, long[] outputStride,
                                                  BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

    private static void broadcastFromLeftAndRight(boolean[] leftBuffer, long[] leftShape, long[] leftStride,
                                                  boolean[] rightBuffer, long[] rightShape, long[] rightStride,
                                                  boolean[] outputBuffer, long[] outputStride,
                                                  BiFunction<Boolean, Boolean, Boolean> op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

    private static void broadcastFromLeftAndRight(Object[] leftBuffer, long[] leftShape, long[] leftStride,
                                                  Object[] rightBuffer, long[] rightShape, long[] rightStride,
                                                  Object[] outputBuffer, long[] outputStride,
                                                  BiFunction op) {

        for (int i = 0; i < outputBuffer.length; i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer[i] = op.apply(leftBuffer[k], rightBuffer[j]);
        }
    }

}
