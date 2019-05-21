package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;

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
        Preconditions.checkArgument(leftBuffer.length >= rightBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = mapBroadcastedIndex(leftStride, rightShape, rightStride, i);

            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[j]);
        }

    }

    private static int mapBroadcastedIndex(long[] fromStride, long[] toShape, long[] toStride, int fromFlatIndex) {

        long[] shapeIndices = new long[fromStride.length];
        long remainder = fromFlatIndex;
        for (int i = 0; i < fromStride.length; i++) {
            shapeIndices[i] = remainder / fromStride[i];
            remainder -= shapeIndices[i] * fromStride[i];
        }

        long[] mappedShapeIndices = new long[shapeIndices.length];

        for (int s = 0; s < shapeIndices.length; s++) {
            mappedShapeIndices[s] = shapeIndices[s] % toShape[s];
        }

        int flatIndex = 0;
        for (int i = 0; i < mappedShapeIndices.length; i++) {

            if (mappedShapeIndices[i] >= toShape[i]) {
                throw new IllegalArgumentException(
                    "Invalid index " + Arrays.toString(mappedShapeIndices) + " for shape " + Arrays.toString(toShape)
                );
            }

            flatIndex += toStride[i] * mappedShapeIndices[i];
        }

        return flatIndex;
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
        Preconditions.checkArgument(leftBuffer.length <= rightBuffer.length);
        for (int i = 0; i < outputBuffer.length; i++) {

            int j = mapBroadcastedIndex(rightStride, leftShape, leftStride, i);

            outputBuffer[i] = op.apply(leftBuffer[j], rightBuffer[i]);
        }

    }


}
