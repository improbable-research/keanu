package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMDoubleTensorBroadcast {

    @AllArgsConstructor
    public static class ResultWrapper<T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> {
        public final B outputBuffer;
        public final long[] outputShape;
        public final long[] outputStride;
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
    ResultWrapper<T, B> broadcastIfNeeded(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                                          B leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                                          B rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                                          BiFunction<T, T, T> op,
                                          boolean inPlace) {

        final boolean needsBroadcast = !Arrays.equals(leftShape, rightShape);

        B outputBuffer;
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

                outputBuffer = inPlace ? leftBuffer : factory.createNew(leftBufferLength);
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
            outputBuffer = inPlace ? leftBuffer : factory.createNew(leftBufferLength);
            outputShape = inPlace ? leftShape : Arrays.copyOf(leftShape, leftShape.length);
            outputStride = inPlace ? leftStride : Arrays.copyOf(leftStride, leftStride.length);

            elementwiseBinaryOp(leftBuffer, rightBuffer, op, outputBuffer);
        }

        return new ResultWrapper<>(outputBuffer, outputShape, outputStride);
    }

    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void scalarLeft(T left, B rightBuffer,
                                                                                     B outputBuffer,
                                                                                     BiFunction<T, T, T> op) {
        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(left, rightBuffer.get(i)), i);
        }
    }


    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void scalarRight(B leftBuffer, T right,
                                                                                      B outputBuffer,
                                                                                      BiFunction<T, T, T> op) {
        for (int i = 0; i < leftBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), right), i);
        }
    }

    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void elementwiseBinaryOp(B leftBuffer,
                                                                                              B rightBuffer,
                                                                                              BiFunction<T, T, T> op,
                                                                                              B outputBuffer) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(i)), i);
        }
    }

    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> ResultWrapper<T, B>
    broadcastBinaryOp(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                      B leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                      B rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                      BiFunction<T, T, T> op,
                      boolean inPlace) {

        final long[] resultShape = Shape.broadcastOutputShape(leftShape, rightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, leftShape);

        final B outputBuffer;
        final long[] outputStride;

        if (resultShapeIsLeftSideShape) {

            outputBuffer = inPlace ? leftBuffer : factory.createNew(leftBufferLength);
            outputStride = leftStride;

            //e.g. [2, 2] * [1, 2]
            broadcastFromRight(
                leftBuffer, leftStride, rightBuffer,
                rightShape, rightStride,
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
                    rightBuffer, rightStride,
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


    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void broadcast(B buffer, long[] shape, long[] stride,
                                                                                   B outputBuffer, long[] outputStride) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {

            int j = getBroadcastedFlatIndex(i, outputStride, shape, stride);

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
    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void broadcastFromRight(B leftBuffer, long[] leftStride,
                                                                                             B rightBuffer, long[] rightShape, long[] rightStride,
                                                                                             B outputBuffer, BiFunction<T, T, T> op) {
        for (int i = 0; i < outputBuffer.getLength(); i++) {

            int j = getBroadcastedFlatIndex(i, leftStride, rightShape, rightStride);
            outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(j)), i);
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
    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void broadcastFromLeft(B leftBuffer, long[] leftShape, long[] leftStride,
                                                                                            B rightBuffer, long[] rightStride,
                                                                                            B outputBuffer, BiFunction<T, T, T> op) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {

            int j = getBroadcastedFlatIndex(i, rightStride, leftShape, leftStride);

            outputBuffer.set(op.apply(leftBuffer.get(j), rightBuffer.get(i)), i);
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

    private static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>> void broadcastFromLeftAndRight(B leftBuffer, long[] leftShape, long[] leftStride,
                                                                                                    B rightBuffer, long[] rightShape, long[] rightStride,
                                                                                                    B outputBuffer, long[] outputStride,
                                                                                                    BiFunction<T, T, T> op) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer.set(op.apply(leftBuffer.get(k), rightBuffer.get(j)), i);
        }
    }

}
