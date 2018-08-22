package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class provides shim methods for the ND4J INDArray class.
 * The INDArray broadcast operations are currently broken
 * (https://github.com/deeplearning4j/deeplearning4j/issues/5893).
 * Until this is fixed in the ND4J codebase, these methods can be
 * used to work around the issue. The need for this should be
 * reevaluated each time the ND4J dependency is updated.
 */
public class INDArrayShim {

    public static INDArray muli(INDArray left, INDArray right, INDArray result) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.muli(right);
        } else {
            return broadcastMultiply(left, right, result);
        }
    }

    private static INDArray broadcastMultiply(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
        if (a.shape().length < b.shape().length) {
            return broadcastMultiply(b, a, result);
        }
        result = a.dup();
        return execBroadcast(a, b,
            new BroadcastMulOp(a, b, result, broadcastDimensions)
        );
    }

    public static INDArray divi(INDArray left, INDArray right, INDArray result) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.divi(right);
        } else {
            return broadcastDivide(left, right, result);
        }
    }

    private static INDArray broadcastDivide(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
        if (a.shape().length < b.shape().length) {
            return broadcastMultiply(b, a.rdiv(1.0), result);
        }
        result = a.dup();
        return execBroadcast(a, b,
            new BroadcastDivOp(a, b, result, broadcastDimensions)
        );
    }

    public static INDArray addi(INDArray left, INDArray right, INDArray result) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.addi(right);
        } else {
            return broadcastPlus(left, right, result);
        }
    }

    private static INDArray broadcastPlus(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
        if (a.shape().length < b.shape().length) {
            return broadcastPlus(b, a, result);
        }
        result = a.dup();
        return execBroadcast(a, b,
            new BroadcastAddOp(a, b, result, broadcastDimensions)
        );
    }

    public static INDArray subi(INDArray left, INDArray right, INDArray result) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.subi(right);
        } else {
            return broadcastMinus(left, right, result);
        }
    }

    private static INDArray broadcastMinus(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = Shape.getBroadcastDimensions(a.shape(), b.shape());
        if (a.shape().length < b.shape().length) {
            return broadcastPlus(a.neg(), b, b.dup());
        } else {
            return execBroadcast(a, b,
                new BroadcastSubOp(a, b, result, broadcastDimensions)
            );
        }
    }

    private static INDArray execBroadcast(INDArray a, INDArray b, BroadcastOp op) {
        int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
        return Nd4j.getExecutioner().exec(op, executeAlong);
    }

    private static int[] getBroadcastDimensions(int[] shapeA, int[] shapeB) {
        int maxRank = Math.max(shapeA.length, shapeB.length);

        if (shapeA.length == shapeB.length) {
            // do nuthing
        } else if (shapeA.length < shapeB.length) {
            shapeA = TensorShape.shapeToDesiredRankByPrependingOnes(shapeA, shapeB.length);
        } else {
            shapeB = TensorShape.shapeToDesiredRankByPrependingOnes(shapeB, shapeA.length);
        }

        List<Integer> along = new ArrayList<>();

        for (int i = maxRank - 1; i >= 0; i--) {
            if (shapeA[i] != shapeB[i]){
                along.add(i);
            }
        }
        return Ints.toArray(along);
    }

    private static int[] getBroadcastAlongDimensions(int[] shapeA, int[] shapeB) {
        int maxRank = Math.max(shapeA.length, shapeB.length);

        if (shapeA.length == shapeB.length) {
            // do nuthing
        } else if (shapeA.length < shapeB.length) {
            shapeA = TensorShape.shapeToDesiredRankByPrependingOnes(shapeA, shapeB.length);
        } else {
            shapeB = TensorShape.shapeToDesiredRankByPrependingOnes(shapeB, shapeA.length);
        }

        List<Integer> along = new ArrayList<>();

        for (int i = maxRank - 1; i >= 0; i--) {
            if (shapeA[i] == shapeB[i]){
                along.add(i);
            }
        }
        return Ints.toArray(along);
    }
}
