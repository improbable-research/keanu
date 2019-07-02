package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.buffer.JVMBuffer;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMTensorBroadcast {

    public static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    ResultWrapper<OUT, OUTBUFFER> broadcastIfNeeded(JVMBuffer.ArrayWrapperFactory<OUT, OUTBUFFER> factory,
                                                    INBUFFER leftBuffer, long[] leftShape, long[] leftStride, long leftBufferLength,
                                                    INBUFFER rightBuffer, long[] rightShape, long[] rightStride, long rightBufferLength,
                                                    BiFunction<IN, IN, OUT> op,
                                                    boolean inPlace) {

        final boolean needsBroadcast = !Arrays.equals(leftShape, rightShape);

        OUTBUFFER outputBuffer;
        long[] outputShape;
        long[] outputStride;

        if (needsBroadcast) {

            //Short circuit for broadcast with scalars
            if (leftShape.length == 0) {

                outputBuffer = factory.createNew(rightBufferLength);
                outputShape = Arrays.copyOf(rightShape, rightShape.length);
                outputStride = Arrays.copyOf(rightStride, rightShape.length);
                scalarLeft(leftBuffer.get(0), rightBuffer, outputBuffer, op);

            } else if (rightShape.length == 0) {

                outputBuffer = inPlace ? (OUTBUFFER) leftBuffer : factory.createNew(leftBufferLength);
                outputShape = inPlace ? leftShape : Arrays.copyOf(leftShape, leftShape.length);
                outputStride = inPlace ? leftStride : Arrays.copyOf(leftStride, leftStride.length);
                scalarRight(leftBuffer, rightBuffer.get(0), outputBuffer, op);

            } else {

                return broadcastBinaryOp(
                    factory,
                    leftBuffer, leftShape, leftStride, leftBufferLength,
                    rightBuffer, rightShape, rightStride, rightBufferLength,
                    op, inPlace
                );
            }

        } else {
            outputBuffer = inPlace ? (OUTBUFFER) leftBuffer : factory.createNew(leftBufferLength);
            outputShape = inPlace ? leftShape : Arrays.copyOf(leftShape, leftShape.length);
            outputStride = inPlace ? leftStride : Arrays.copyOf(leftStride, leftStride.length);

            elementwiseBinaryOp(leftBuffer, rightBuffer, op, outputBuffer);
        }

        return new ResultWrapper<>(outputBuffer, outputShape, outputStride);
    }

    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void scalarLeft(IN left, INBUFFER rightBuffer,
                    OUTBUFFER outputBuffer,
                    BiFunction<IN, IN, OUT> op) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(left, rightBuffer.get(i)), i);
        }
    }


    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void scalarRight(INBUFFER leftBuffer, IN right,
                     OUTBUFFER outputBuffer,
                     BiFunction<IN, IN, OUT> op) {
        for (int i = 0; i < leftBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), right), i);
        }
    }

    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void elementwiseBinaryOp(INBUFFER leftBuffer,
                             INBUFFER rightBuffer,
                             BiFunction<IN, IN, OUT> op,
                             OUTBUFFER outputBuffer) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(i)), i);
        }
    }

    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    ResultWrapper<OUT, OUTBUFFER> broadcastBinaryOp(JVMBuffer.ArrayWrapperFactory<OUT, OUTBUFFER> factory,
                                                    INBUFFER leftBuffer, long[] leftShape, long[] leftStride, long leftBufferLength,
                                                    INBUFFER rightBuffer, long[] rightShape, long[] rightStride, long rightBufferLength,
                                                    BiFunction<IN, IN, OUT> op,
                                                    boolean inPlace) {

        final long[] resultShape = TensorShape.getBroadcastResultShape(leftShape, rightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, leftShape);

        final OUTBUFFER outputBuffer;
        final long[] outputStride;

        if (resultShapeIsLeftSideShape) {

            outputBuffer = inPlace ? (OUTBUFFER) leftBuffer : factory.createNew(leftBufferLength);
            outputStride = leftStride;

            //e.g. [2, 2] * [1, 2]
            broadcastFromRight(
                leftBuffer, leftShape, leftStride,
                rightBuffer, rightShape, rightStride,
                outputBuffer, op
            );

        } else {

            final boolean resultShapeIsRightSideShape = Arrays.equals(resultShape, rightShape);

            if (resultShapeIsRightSideShape) {

                outputBuffer = factory.createNew(rightBufferLength);
                outputStride = rightStride;

                //e.g. [2] / [2, 2]
                broadcastFromLeft(
                    leftBuffer, leftShape, leftStride,
                    rightBuffer, rightShape, rightStride,
                    outputBuffer, op
                );

            } else {

                outputBuffer = factory.createNew(getLengthAsInt(resultShape));
                outputStride = getRowFirstStride(resultShape);

                //e.g. [2, 2, 1] * [1, 2, 2]
                broadcastFromLeftAndRight(
                    leftBuffer, leftShape, leftStride,
                    rightBuffer, rightShape, rightStride,
                    outputBuffer, outputStride, op
                );
            }
        }

        return new ResultWrapper<>(outputBuffer, resultShape, outputStride);
    }


    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> void broadcast(B buffer, long[] shape, long[] stride,
                                                                                      B outputBuffer, long[] outputStride) {

        for (long i = 0; i < outputBuffer.getLength(); i++) {

            long j = getBroadcastedFlatIndex(i, outputStride, shape, stride);

            outputBuffer.set(buffer.get(j), i);
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
    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void broadcastFromRight(INBUFFER leftBuffer, long[] leftShape, long[] leftStride,
                            INBUFFER rightBuffer, long[] rightShape, long[] rightStride,
                            OUTBUFFER outputBuffer, BiFunction<IN, IN, OUT> op) {


        if (canQuickBroadcast(rightShape, leftShape)) {
            for (int i = 0; i < outputBuffer.getLength(); i++) {

                long j = i % rightBuffer.getLength();
                outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(j)), i);
            }
        } else {
            for (long i = 0; i < outputBuffer.getLength(); i++) {

                long j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);
                outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(j)), i);
            }
        }

    }

    private static boolean canQuickBroadcast(long[] fromShape, long[] broadcastShape) {

        boolean b = true;

        for (int i = 1; i <= fromShape.length; i++) {
            if (fromShape[fromShape.length - i] != broadcastShape[broadcastShape.length - i]) {
                b = false;
            } else {
                if (!b) {
                    return false;
                }
            }
        }

        return true;
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
    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void broadcastFromLeft(INBUFFER leftBuffer, long[] leftShape, long[] leftStride,
                           INBUFFER rightBuffer, long[] rightShape, long[] rightStride,
                           OUTBUFFER outputBuffer, BiFunction<IN, IN, OUT> op) {


        if (canQuickBroadcast(leftShape, rightShape)) {
            for (long i = 0; i < outputBuffer.getLength(); i++) {

                long j = i % leftBuffer.getLength();
                outputBuffer.set(op.apply(leftBuffer.get(j), rightBuffer.get(i)), i);
            }
        } else {
            for (long i = 0; i < outputBuffer.getLength(); i++) {

                long j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

                outputBuffer.set(op.apply(leftBuffer.get(j), rightBuffer.get(i)), i);
            }
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

    private static <IN, OUT, INBUFFER extends JVMBuffer.PrimitiveArrayWrapper<IN, INBUFFER>, OUTBUFFER extends JVMBuffer.PrimitiveArrayWrapper<OUT, OUTBUFFER>>
    void broadcastFromLeftAndRight(INBUFFER leftBuffer, long[] leftShape, long[] leftStride,
                                   INBUFFER rightBuffer, long[] rightShape, long[] rightStride,
                                   OUTBUFFER outputBuffer, long[] outputStride,
                                   BiFunction<IN, IN, OUT> op) {

        for (long i = 0; i < outputBuffer.getLength(); i++) {

            long k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            long j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer.set(op.apply(leftBuffer.get(k), rightBuffer.get(j)), i);
        }
    }

}
