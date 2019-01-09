package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldGreaterThan;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldGreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldLessThan;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldLessThanOrEqual;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TypedINDArrayFactory.valueArrayOf;

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

    /*
     * We need to load ND4J in a separate thread as on load it sets the FTZ and DAZ flags in the processor for the
     * thread that does the load.  This causes issues with Apache Math that makes use of Sub-normal values (in
     * particular to initialisation values for the BrentOptimizer).
     *
     * We have raised https://github.com/deeplearning4j/deeplearning4j/issues/6690 to address this
     */
    public static void startNewThreadForNd4j() {
        Thread nd4jInitThread = new Thread(() -> Nd4j.create(1));
        nd4jInitThread.start();
        try {
            nd4jInitThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static INDArray applyInlineOperation(INDArray left,
                                                 INDArray right,
                                                 Function<INDArray, INDArray> inverseOperand,
                                                 BiFunction<INDArray, INDArray, INDArray> baseInlineOp,
                                                 BiFunction<INDArray, INDArray, INDArray> inverseInlineOp,
                                                 QuadFunction<INDArray, INDArray, INDArray, List<Integer>, INDArray> baseBroadcastOp,
                                                 QuadFunction<INDArray, INDArray, INDArray, List<Integer>, INDArray> inverseBroadcastOp) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return baseInlineOp.apply(left, right);
        } else if (left.length() == 1) {
            return applyScalarTensorOperationWithPreservedShape(inverseOperand.apply(right), left, inverseInlineOp);
        } else if (right.length() == 1) {
            return applyScalarTensorOperationWithPreservedShape(left, right, baseInlineOp);
        } else if (shapeAIsSmallerThanShapeB(left.shape(), right.shape())) {
            return applyBroadcastOperation(inverseOperand.apply(right), left, inverseBroadcastOp);
        } else {
            return applyBroadcastOperation(left, right, baseBroadcastOp);
        }
    }

    private static INDArray applyBroadcastOperation(INDArray left,
                                                    INDArray right,
                                                    QuadFunction<INDArray, INDArray, INDArray, List<Integer>, INDArray> baseBroadcastOp) {
        List<Integer> broadcastDimensions = getBroadcastDimensions(left.shape(), right.shape());
        INDArray result = Nd4j.create(Shape.broadcastOutputShape(left.shape(), right.shape()));
        return baseBroadcastOp.apply(left, right, result, broadcastDimensions);
    }

    public static INDArray muli(INDArray left, INDArray right) {
        return applyInlineOperation(
            left, right,
            a -> a,
            INDArray::muli,
            INDArray::muli,
            (l, r, result, dims) -> Broadcast.mul(l, r, result, Ints.toArray(dims)),
            (l, r, result, dims) -> Broadcast.mul(l, r, result, Ints.toArray(dims)));
    }

    public static INDArray divi(INDArray left, INDArray right) {
        return applyInlineOperation(
            left, right,
            a -> a.rdiv(1.),
            INDArray::divi,
            INDArray::muli,
            (l, r, result, dims) -> Broadcast.div(l, r, result, Ints.toArray(dims)),
            (l, r, result, dims) -> Broadcast.mul(l, r, result, Ints.toArray(dims)));
    }

    public static INDArray addi(INDArray left, INDArray right) {
        return applyInlineOperation(
            left, right,
            a -> a,
            INDArray::addi,
            INDArray::addi,
            (l, r, result, dims) -> Broadcast.add(l, r, result, Ints.toArray(dims)),
            (l, r, result, dims) -> Broadcast.add(l, r, result, Ints.toArray(dims)));
    }

    public static INDArray subi(INDArray left, INDArray right) {
        return applyInlineOperation(
            left, right,
            a -> a.neg(),
            INDArray::subi,
            INDArray::addi,
            (l, r, result, dims) -> Broadcast.sub(l, r, result, Ints.toArray(dims)),
            (l, r, result, dims) -> Broadcast.add(l, r, result, Ints.toArray(dims)));
    }

    private static INDArray applyScalarTensorOperationWithPreservedShape(INDArray tensor, INDArray scalarTensor, BiFunction<INDArray, INDArray, INDArray> operation) {
        INDArray result = operation.apply(tensor, scalarTensor.getScalar(0));
        long[] resultShape = Shape.broadcastOutputShape(tensor.shape(), scalarTensor.shape());
        return result.reshape(resultShape);
    }

    public static INDArray pow(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, Transforms::pow);
    }

    public static INDArray max(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, Transforms::max);
    }

    public static INDArray min(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, Transforms::min);
    }

    public static INDArray atan2(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, Transforms::atan2);
    }

    public static INDArray lt(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, INDArray::lt);
    }

    public static INDArray gt(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, INDArray::gt);
    }

    public static INDArray eq(INDArray left, INDArray right) {
        return performOperationWithScalarTensorPreservingShape(left, right, INDArray::eq);
    }

    private static INDArray performOperationWithScalarTensorPreservingShape(INDArray left, INDArray right, BiFunction<INDArray, INDArray, INDArray> operation) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result = (left.length() == 1) ?
                operation.apply(Nd4j.valueArrayOf(right.shape(), left.getDouble(0)), right) :
                operation.apply(left, Nd4j.valueArrayOf(left.shape(), right.getDouble(0)));
            return result.reshape(resultShape);
        } else {
            return operation.apply(left, right);
        }
    }

    @FunctionalInterface
    interface QuadFunction<First, Second, Third, Fourth, Result> {
        Result apply(First one, Second two, Third three, Fourth four);
    }

    private static INDArray executeNd4jTransformOpWithPreservedScalarTensorShape(INDArray mask, INDArray right, DataBuffer.Type bufferType, QuadFunction<INDArray, INDArray, INDArray, Long, BaseTransformOp> baseTransformOpConstructor) {
        if (mask.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(mask.shape(), right.shape());
            if (mask.length() == 1) {
                mask = Nd4j.valueArrayOf(right.shape(), mask.getDouble(0));
                Nd4j.getExecutioner().exec(
                    baseTransformOpConstructor.apply(mask, right, mask, mask.length())
                );
            } else {
                Nd4j.getExecutioner().exec(
                    baseTransformOpConstructor.apply(mask,
                        valueArrayOf(mask.shape(), right.getDouble(0), bufferType),
                        mask,
                        mask.length()
                    )
                );
            }
            return mask.reshape(resultShape);
        } else {
            Nd4j.getExecutioner().exec(
                baseTransformOpConstructor.apply(mask, right, mask, mask.length())
            );
            return mask;
        }
    }

    public static INDArray getGreaterThanMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        return executeNd4jTransformOpWithPreservedScalarTensorShape(mask, right, bufferType, OldGreaterThan::new);
    }

    public static INDArray getGreaterThanOrEqualToMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        return executeNd4jTransformOpWithPreservedScalarTensorShape(mask, right, bufferType, OldGreaterThanOrEqual::new);
    }

    public static INDArray getLessThanMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        return executeNd4jTransformOpWithPreservedScalarTensorShape(mask, right, bufferType, OldLessThan::new);
    }

    public static INDArray getLessThanOrEqualToMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        return executeNd4jTransformOpWithPreservedScalarTensorShape(mask, right, bufferType, OldLessThanOrEqual::new);
    }

    private static boolean shapeAIsSmallerThanShapeB(long[] shapeA, long[] shapeB) {
        if (shapeA.length == shapeB.length) {
            for (int ind = 0; ind < shapeA.length; ind++) {
                if (shapeA[ind] < shapeB[ind]) {
                    return true;
                }
            }
        }
        return shapeA.length < shapeB.length;
    }

    private static List<Integer> getBroadcastDimensions(long[] shapeA, long[] shapeB) {
        int minRank = Math.min(shapeA.length, shapeB.length);
        int maxRank = Math.max(shapeA.length, shapeB.length);

        List<Integer> along = new ArrayList<>();

        for (int i = minRank - 1; i >= 0; i--) {
            if (shapeA[shapeA.length - i - 1] == shapeB[shapeB.length - i - 1]) {
                along.add(maxRank - i - 1);
            }
        }
        return along;
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

                //dup is required before reshaping due to a bug in ND4J that doesn't always correctly
                //duplicate true vectors.
                return result.dup().reshape(newShape);
            } else {
                return result;
            }
        }
    }

}
