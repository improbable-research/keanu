package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;

import java.util.Arrays;
import java.util.function.BiFunction;

public class JVMDoubleTensorBroadcast {

    public enum BroadcastableDoubleOperation implements BiFunction<Double, Double, Double> {

        ADD {
            @Override
            public Double apply(Double left, Double right) {
                return left + right;
            }
        },

        SUB {
            @Override
            public Double apply(Double left, Double right) {
                return left - right;
            }
        },

        MUL {
            @Override
            public Double apply(Double left, Double right) {
                return left * right;
            }
        },


        DIV {
            @Override
            public Double apply(Double left, Double right) {
                return left / right;
            }
        },

        GT_MASK {
            @Override
            public Double apply(Double left, Double right) {
                return left > right ? 1.0 : 0.0;
            }
        },

        GTE_MASK {
            @Override
            public Double apply(Double left, Double right) {
                return left >= right ? 1.0 : 0.0;
            }
        },

        LT_MASK {
            @Override
            public Double apply(Double left, Double right) {
                return left < right ? 1.0 : 0.0;
            }
        },

        LTE_MASK {
            @Override
            public Double apply(Double left, Double right) {
                return left <= right ? 1.0 : 0.0;
            }
        }

    }

    static void elementwise(double[] leftBuffer, double[] rightBuffer, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }

    }

    static void scalarLeft(double left, double[] rightBuffer, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(left, rightBuffer[i]);
        }

    }

    static void scalarRight(double[] leftBuffer, double right, double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], right);
        }

    }

    /**
     * Right buffer is shorter than left
     *
     * @param leftBuffer
     * @param leftShape
     * @param leftStride
     * @param rightBuffer
     * @param rightShape
     * @param rightStride
     * @param outputBuffer
     * @param op
     * @return
     */
    static void broadcastFromRight(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                   double[] rightBuffer, long[] rightShape, long[] rightStride,
                                   double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        //implicitly pad lower ranks with 1s. E.g. [3, 3] & [3] -> [3, 3] -> [1, 3]
        final long[] paddedRightShape;
        final long[] paddedRightStride;

        if (leftShape.length != rightShape.length) {
            paddedRightShape = TensorShape.shapeToDesiredRankByPrependingOnes(rightShape, leftShape.length);
            paddedRightStride = TensorShape.getRowFirstStride(paddedRightShape);
        } else {
            paddedRightShape = rightShape;
            paddedRightStride = rightStride;
        }

        for (int i = 0; i < outputBuffer.length; i++) {

            long[] shapeIndices = TensorShape.getShapeIndices(leftShape, leftStride, i);

            long[] mappedShapeIndices = new long[shapeIndices.length];

            for (int s = 0; s < shapeIndices.length; s++) {
                mappedShapeIndices[s] = shapeIndices[s] % paddedRightShape[s];
            }

            int j = Ints.checkedCast(TensorShape.getFlatIndex(paddedRightShape, paddedRightStride, mappedShapeIndices));

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }

    }

    /**
     * Left buffer is shorter than right
     *
     * @param leftBuffer
     * @param leftShape
     * @param leftStride
     * @param rightBuffer
     * @param rightShape
     * @param rightStride
     * @param outputBuffer
     * @param op
     * @return
     */
    static void broadcastFromLeft(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                  double[] rightBuffer, long[] rightShape, long[] rightStride,
                                  double[] outputBuffer, BiFunction<Double, Double, Double> op) {

        //implicitly pad lower ranks with 1s. E.g. [3, 3] & [3] -> [3, 3] -> [1, 3]
        final long[] paddedLeftShape;
        final long[] paddedLeftStride;

        if (leftShape.length != rightShape.length) {
            paddedLeftShape = TensorShape.shapeToDesiredRankByPrependingOnes(leftShape, rightShape.length);
            paddedLeftStride = TensorShape.getRowFirstStride(paddedLeftShape);
        } else {
            paddedLeftShape = leftShape;
            paddedLeftStride = leftStride;
        }

        for (int i = 0; i < outputBuffer.length; i++) {

            long[] shapeIndices = TensorShape.getShapeIndices(rightShape, rightStride, i);

            long[] mappedShapeIndices = new long[shapeIndices.length];

            for (int s = 0; s < shapeIndices.length; s++) {
                mappedShapeIndices[s] = shapeIndices[s] % paddedLeftShape[s];
            }

            int j = Ints.checkedCast(TensorShape.getFlatIndex(paddedLeftShape, paddedLeftStride, mappedShapeIndices));

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }

    }


}
