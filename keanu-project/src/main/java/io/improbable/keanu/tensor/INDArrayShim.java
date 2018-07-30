package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    public static INDArray muli(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            left.muli(right);
        } else {
            broadcastMultiply(left, right, left);
        }
        return left;
    }

    public static void broadcastMultiply(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = Shape.getBroadcastDimensions(a.shape(), b.shape());
        int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
        Nd4j.getExecutioner().exec(
            new BroadcastMulOp(a, b, result, broadcastDimensions),
            executeAlong
        );
    }

    public static INDArray divi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            left.divi(right);
        } else {
            broadcastDivide(left, right, left);
        }
        return left;
    }

    public static void broadcastDivide(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = Shape.getBroadcastDimensions(a.shape(), b.shape());
        int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
        Nd4j.getExecutioner().exec(
            new BroadcastDivOp(a, b, result, broadcastDimensions),
            executeAlong
        );
    }

    public static INDArray addi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            left.addi(right);
        } else {
            broadcastPlus(left, right, left);
        }
        return left;
    }

    public static void broadcastPlus(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = Shape.getBroadcastDimensions(a.shape(), b.shape());
        int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
        Nd4j.getExecutioner().exec(
            new BroadcastAddOp(a, b, result, broadcastDimensions),
            executeAlong
        );
    }

    public static INDArray subi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            left.subi(right);
        } else {
            broadcastMinus(left, right, left);
        }
        return left;
    }

    public static void broadcastMinus(INDArray a, INDArray b, INDArray result) {
        int[] broadcastDimensions = Shape.getBroadcastDimensions(a.shape(), b.shape());
        int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
        Nd4j.getExecutioner().exec(
            new BroadcastSubOp(a, b, result, broadcastDimensions),
            executeAlong
        );
    }

    private static int[] getBroadcastAlongDimensions(int[] shapeA, int[] shapeB) {
        int minRank = Math.min(shapeA.length, shapeB.length);
        List<Integer> along = new ArrayList<>();
        for (int i = 0; i < minRank; i++) {
            if (shapeA[i] == shapeB[i]) {
                along.add(i);
            }
        }
        return Ints.toArray(along);
    }
}
