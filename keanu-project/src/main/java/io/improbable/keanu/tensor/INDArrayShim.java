package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Broadcast;
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
 * <p>
 * To work around another issue in ND4J where you cannot broadcast
 * a higher rank tensor onto a lower rank tensor, the shim broadcast operations
 * ensure the higher rank tensor is always being operated on. In the case of
 * subtract and minus, this requires a small change in the logic, as A - B != B - A and
 * A / B != B / A.
 */
public class INDArrayShim {

    public static INDArray muli(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.muli(right);
        } else {
            return broadcastMultiply(left, right);
        }
    }

    private static INDArray broadcastMultiply(INDArray a, INDArray b) {
        if (a.shape().length < b.shape().length) {
            return broadcastMultiply(b, a);
        } else {
            int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
            INDArray result = Nd4j.create(Shape.broadcastOutputShape(a.shape(), b.shape()));
            return Broadcast.mul(a, b, result, broadcastDimensions);
        }
    }

    public static INDArray divi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.divi(right);
        } else {
            return broadcastDivide(left, right);
        }
    }

    private static INDArray broadcastDivide(INDArray a, INDArray b) {
        if (a.shape().length < b.shape().length) {
            return broadcastMultiply(b.rdiv(1.0), a);
        } else {
            int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
            INDArray result = Nd4j.create(Shape.broadcastOutputShape(a.shape(), b.shape()));
            return Broadcast.div(a, b, result, broadcastDimensions);
        }
    }

    public static INDArray addi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.addi(right);
        } else {
            return broadcastPlus(left, right);
        }
    }

    private static INDArray broadcastPlus(INDArray a, INDArray b) {
        if (a.shape().length < b.shape().length) {
            return broadcastPlus(b, a);
        } else {
            int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
            INDArray result = Nd4j.create(Shape.broadcastOutputShape(a.shape(), b.shape()));
            return Broadcast.add(a, b, result, broadcastDimensions);
        }
    }

    public static INDArray subi(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.subi(right);
        } else {
            return broadcastMinus(left, right);
        }
    }

    private static INDArray broadcastMinus(INDArray a, INDArray b) {
        if (a.shape().length < b.shape().length) {
            return broadcastPlus(a, b.neg());
        } else {
            int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
            INDArray result = Nd4j.create(Shape.broadcastOutputShape(a.shape(), b.shape()));
            return Broadcast.sub(a, b, result, broadcastDimensions);
        }
    }

    private static int[] getBroadcastDimensions(long[] shapeA, long[] shapeB) {
        int minRank = Math.min(shapeA.length, shapeB.length);
        int maxRank = Math.max(shapeA.length, shapeB.length);

        List<Integer> along = new ArrayList<>();

        for (int i = minRank - 1; i >= 0; i--) {
            if (shapeA[shapeA.length - i - 1] == shapeB[shapeB.length - i - 1]) {
                along.add(maxRank - i - 1);
            }
        }
        return Ints.toArray(along);
    }

    public static INDArray sum(INDArray tensor, int... overDimensions) {
        overDimensions = TensorShape.getAbsoluteDimensions(tensor.rank(), overDimensions);

        long[] newShape = ArrayUtils.removeAll(tensor.shape(), overDimensions);
        INDArray result = tensor.sum(overDimensions);

        return result.reshape(newShape);
    }

    public static INDArray slice(INDArray tensor, int dimension, long index) {
        if (tensor.rank() <= 1) {
            return tensor.getScalar(index);
        } else {

            INDArray result = tensor.slice(index, dimension);
            if (tensor.rank() == 2) {
                long[] newShape = ArrayUtils.removeAll(tensor.shape(), dimension);

                //dup is required before reshaping due to a but in ND4J that doesn't always correctly
                //duplicate true vectors.
                return result.dup().reshape(newShape);
            } else {
                return result;
            }
        }

    }
}
