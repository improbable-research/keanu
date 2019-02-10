package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.TensorShape;

import java.util.Arrays;

public class JVMDoubleTensorBroadcast {

    public enum BroadcastableDoubleOperation {

        ADD {
            @Override
            double op(double left, double right) {
                return left + right;
            }
        },

        SUB {
            @Override
            double op(double left, double right) {
                return left - right;
            }
        },

        MUL {
            @Override
            double op(double left, double right) {
                return left * right;
            }
        },


        DIV {
            @Override
            double op(double left, double right) {
                return left / right;
            }
        },

        GT_MASK {
            @Override
            double op(double left, double right) {
                return left > right ? 1.0 : 0.0;
            }
        },

        GTE_MASK {
            @Override
            double op(double left, double right) {
                return left >= right ? 1.0 : 0.0;
            }
        },

        LT_MASK {
            @Override
            double op(double left, double right) {
                return left < right ? 1.0 : 0.0;
            }
        },

        LTE_MASK {
            @Override
            double op(double left, double right) {
                return left <= right ? 1.0 : 0.0;
            }
        };

        abstract double op(double left, double right);
    }


    static double[] opWithAutoBroadcast(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                        DoubleTensor right, double[] outputBuffer, BroadcastableDoubleOperation op) {

        final double[] rightBuffer = right.asFlatDoubleArray();
        final long[] rightShape = right.getShape();

        if (Arrays.equals(leftShape, rightShape)) {

            return elementwise(leftBuffer, rightBuffer, outputBuffer, op);
        } else {

            //Short circuit for broadcast with scalars
            if (leftShape.length == 0) {
                return scalarLeft(leftBuffer[0], rightBuffer, outputBuffer, op);
            } else if (rightShape.length == 0) {
                return scalarRight(leftBuffer, rightBuffer[0], outputBuffer, op);
            }

            final long[] rightStride = right.getStride();
            //Allow broadcasting from left and right
            if (leftShape.length > rightShape.length || leftBuffer.length > rightBuffer.length) {
                //e.g. [2, 2] * [1, 2]
                return broadcastFromRight(leftBuffer, leftShape, leftStride, rightBuffer, rightShape, rightStride, outputBuffer, op);
            } else {
                //e.g. [2] / [2, 2]
                return broadcastFromLeft(leftBuffer, leftShape, leftStride, rightBuffer, rightShape, rightStride, outputBuffer, op);
            }
        }
    }

    private static double[] elementwise(double[] leftBuffer, double[] rightBuffer, double[] outputBuffer, BroadcastableDoubleOperation op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.op(leftBuffer[i], rightBuffer[i]);
        }

        return outputBuffer;
    }

    private static double[] scalarLeft(double left, double[] rightBuffer, double[] outputBuffer, BroadcastableDoubleOperation op) {

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.op(left, rightBuffer[i]);
        }

        return outputBuffer;
    }

    private static double[] scalarRight(double[] leftBuffer, double right, double[] outputBuffer, BroadcastableDoubleOperation op) {

        for (int i = 0; i < leftBuffer.length; i++) {
            outputBuffer[i] = op.op(leftBuffer[i], right);
        }

        return outputBuffer;
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
    private static double[] broadcastFromRight(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                               double[] rightBuffer, long[] rightShape, long[] rightStride,
                                               double[] outputBuffer, BroadcastableDoubleOperation op) {

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

            int j = (int) TensorShape.getFlatIndex(paddedRightShape, paddedRightStride, mappedShapeIndices);

            outputBuffer[i] = op.op(leftBuffer[i], rightBuffer[j]);
        }

        return outputBuffer;
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
    private static double[] broadcastFromLeft(double[] leftBuffer, long[] leftShape, long[] leftStride,
                                              double[] rightBuffer, long[] rightShape, long[] rightStride,
                                              double[] outputBuffer, BroadcastableDoubleOperation op) {

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

            int j = (int) TensorShape.getFlatIndex(paddedLeftShape, paddedLeftStride, mappedShapeIndices);

            outputBuffer[i] = op.op(leftBuffer[j], rightBuffer[i]);
        }

        return outputBuffer;
    }


}
