package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.JVMBuffer;
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
        public final JVMBuffer.PrimitiveDoubleWrapper outputBuffer;
        public final long[] outputShape;
        public final long[] outputStride;
    }

    public static ResultWrapper broadcastIfNeeded(JVMBuffer.DoubleArrayWrapperFactory factory,
                                                  JVMBuffer.PrimitiveDoubleWrapper leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                                                  JVMBuffer.PrimitiveDoubleWrapper rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                                                  BiFunction<Double, Double, Double> op,
                                                  boolean inPlace) {
        final boolean needsBroadcast = !Arrays.equals(leftShape, rightShape);

        JVMBuffer.PrimitiveDoubleWrapper outputBuffer;
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

        return new ResultWrapper(outputBuffer, outputShape, outputStride);
    }

    private static void scalarLeft(Double left, JVMBuffer.PrimitiveDoubleWrapper rightBuffer,
                                   JVMBuffer.PrimitiveDoubleWrapper outputBuffer,
                                   BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(left, rightBuffer.get(i)), i);
        }
    }


    private static void scalarRight(JVMBuffer.PrimitiveDoubleWrapper leftBuffer, Double right,
                                    JVMBuffer.PrimitiveDoubleWrapper outputBuffer,
                                    BiFunction<Double, Double, Double> op) {
        for (int i = 0; i < leftBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), right), i);
        }
    }

    private static void elementwiseBinaryOp(JVMBuffer.PrimitiveDoubleWrapper leftBuffer, JVMBuffer.PrimitiveDoubleWrapper rightBuffer,
                                            BiFunction<Double, Double, Double> op,
                                            JVMBuffer.PrimitiveDoubleWrapper outputBuffer) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {
            outputBuffer.set(op.apply(leftBuffer.get(i), rightBuffer.get(i)), i);
        }
    }

    private static ResultWrapper broadcastBinaryOp(JVMBuffer.DoubleArrayWrapperFactory factory,
                                                   JVMBuffer.PrimitiveDoubleWrapper leftBuffer, long[] leftShape, long[] leftStride, int leftBufferLength,
                                                   JVMBuffer.PrimitiveDoubleWrapper rightBuffer, long[] rightShape, long[] rightStride, int rightBufferLength,
                                                   BiFunction op,
                                                   boolean inPlace) {

        final long[] resultShape = Shape.broadcastOutputShape(leftShape, rightShape);
        final boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, leftShape);

        final JVMBuffer.PrimitiveDoubleWrapper outputBuffer;
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

        return new ResultWrapper(outputBuffer, resultShape, outputStride);
    }


    public static void broadcast(JVMBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride,
                                 JVMBuffer.PrimitiveDoubleWrapper outputBuffer, long[] outputStride) {

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
    private static void broadcastFromRight(JVMBuffer.PrimitiveDoubleWrapper leftBuffer, long[] leftStride,
                                           JVMBuffer.PrimitiveDoubleWrapper rightBuffer, long[] rightShape, long[] rightStride,
                                           JVMBuffer.PrimitiveDoubleWrapper outputBuffer, BiFunction<Double, Double, Double> op) {
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
    private static void broadcastFromLeft(JVMBuffer.PrimitiveDoubleWrapper leftBuffer, long[] leftShape, long[] leftStride,
                                          JVMBuffer.PrimitiveDoubleWrapper rightBuffer, long[] rightStride,
                                          JVMBuffer.PrimitiveDoubleWrapper outputBuffer, BiFunction<Double, Double, Double> op) {

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

    private static void broadcastFromLeftAndRight(JVMBuffer.PrimitiveDoubleWrapper leftBuffer, long[] leftShape, long[] leftStride,
                                                  JVMBuffer.PrimitiveDoubleWrapper rightBuffer, long[] rightShape, long[] rightStride,
                                                  JVMBuffer.PrimitiveDoubleWrapper outputBuffer, long[] outputStride,
                                                  BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.getLength(); i++) {

            int k = getBroadcastedFlatIndex(i, outputStride, leftShape, leftStride);
            int j = getBroadcastedFlatIndex(i, outputStride, rightShape, rightStride);

            outputBuffer.set(op.apply(leftBuffer.get(k), rightBuffer.get(j)), i);
        }
    }

}
