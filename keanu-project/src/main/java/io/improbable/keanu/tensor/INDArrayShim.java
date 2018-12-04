package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    private static final DataBuffer.Type BUFFER_TYPE = DataBuffer.Type.DOUBLE;

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

    public static INDArray muli(INDArray left, INDArray right) {
        if (Arrays.equals(left.shape(), right.shape())) {
            return left.muli(right);
        } else if (left.length() == 1 || right.length() == 1) {
            return scalarMultiplyWithPreservedShape(left, right);
        } else {
            return broadcastMultiply(left, right);
        }
    }

    private static INDArray scalarMultiplyWithPreservedShape(INDArray a, INDArray b) {
        if (a.length() != 1) {
            return scalarMultiplyWithPreservedShape(b, a);
        }
        INDArray result = b.muli(a.getScalar(0));
        long[] resultShape = Shape.broadcastOutputShape(a.shape(), b.shape());
        return result.reshape(resultShape);
    }

    private static INDArray broadcastMultiply(INDArray a, INDArray b) {
        if (shapeAIsSmallerThanShapeB(a.shape(), b.shape())) {
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
        }  else if (left.length() == 1 || right.length() == 1) {
        return scalarDivideWithPreservedShape(left, right);
    } else {
        return broadcastDivide(left, right);
    }
}

    private static INDArray scalarDivideWithPreservedShape(INDArray a, INDArray b) {
        if (b.length() != 1) {
            return scalarMultiplyWithPreservedShape(a, b.rdiv(1.0));
        }
        INDArray result = a.divi(b.getScalar(0));
        long[] resultShape = Shape.broadcastOutputShape(a.shape(), b.shape());
        return result.reshape(resultShape);
    }

    private static INDArray broadcastDivide(INDArray a, INDArray b) {
        if (shapeAIsSmallerThanShapeB(a.shape(), b.shape())) {
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
        } else if (left.length() == 1 || right.length() == 1) {
        return scalarAdditionWithPreservedShape(left, right);
    } else {
        return broadcastPlus(left, right);
    }
}

    private static INDArray scalarAdditionWithPreservedShape(INDArray a, INDArray b) {
        if (a.length() != 1) {
            return scalarAdditionWithPreservedShape(b, a);
        }
        INDArray result = b.addi(a.getScalar(0));
        long[] resultShape = Shape.broadcastOutputShape(a.shape(), b.shape());
        return result.reshape(resultShape);
    }

    private static INDArray broadcastPlus(INDArray a, INDArray b) {
        if (shapeAIsSmallerThanShapeB(a.shape(), b.shape())) {
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
        } else if (left.length() == 1 || right.length() == 1) {
            return scalarSubtractionWithPreservedShape(left, right);
        } else {
            return broadcastMinus(left, right);
        }
    }

    private static INDArray scalarSubtractionWithPreservedShape(INDArray a, INDArray b) {
        if (b.length() != 1) {
            return scalarAdditionWithPreservedShape(a, b.neg());
        }
        INDArray result = a.subi(b.getScalar(0));
        long[] resultShape = Shape.broadcastOutputShape(a.shape(), b.shape());
        return result.reshape(resultShape);
    }

    private static INDArray broadcastMinus(INDArray a, INDArray b) {
        if (shapeAIsSmallerThanShapeB(a.shape(), b.shape())) {
            return broadcastPlus(a, b.neg());
        } else {
            int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
            INDArray result = Nd4j.create(Shape.broadcastOutputShape(a.shape(), b.shape()));
            return Broadcast.sub(a, b, result, broadcastDimensions);
        }
    }

    public static INDArray pow(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Transforms.pow(Nd4j.valueArrayOf(right.shape(), left.getDouble(0)), right, false);
            } else {
                result = Transforms.pow(left, right.getDouble(0), false);
            }
            return result.reshape(resultShape);
        }
        return Transforms.pow(left, right, false);
    }

    public static INDArray max(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Transforms.max(Nd4j.valueArrayOf(right.shape(), left.getDouble(0)), right, false);
            } else {
                result = Transforms.max(left, right.getDouble(0), false);
            }
            return result.reshape(resultShape);
        }
        return Transforms.max(left, right, false);
    }

    public static INDArray min(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Transforms.min(Nd4j.valueArrayOf(right.shape(), left.getDouble(0)), right, false);
            } else {
                result = Transforms.min(left, right.getDouble(0), false);
            }
            return result.reshape(resultShape);
        }
        return Transforms.min(left, right, false);
    }

    public static INDArray lt(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Nd4j.valueArrayOf(right.shape(), left.getDouble(0)).lt(right);
            } else {
                result = left.lt(right.getDouble(0));
            }
            return result.reshape(resultShape);
        }
        return left.lt(right);
    }

    public static INDArray gt(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Nd4j.valueArrayOf(right.shape(), left.getDouble(0)).gt(right);
            } else {
                result = left.gt(right.getDouble(0));
            }
            return result.reshape(resultShape);
        }
        return left.gt(right);
    }

    public static INDArray lessThanOrEqual(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                left = Nd4j.valueArrayOf(right.shape(), left.getDouble(0));
                Nd4j.getExecutioner().exec(new OldLessThanOrEqual(left, right, left, left.length()));
                result = left;
            } else {
                result = left.lte(right.getDouble(0));
            }
            return result.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(new OldLessThanOrEqual(left, right, left, left.length()));
        return left;
    }

    public static INDArray greaterThanOrEqual(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                left = Nd4j.valueArrayOf(right.shape(), left.getDouble(0));
                Nd4j.getExecutioner().exec(new OldGreaterThanOrEqual(left, right, left, left.length()));
                result = left;
            } else {
                result = left.gte(right.getDouble(0));
            }
            return result.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(new OldGreaterThanOrEqual(left, right, left, left.length()));
        return left;
    }

    public static INDArray eq(INDArray left, INDArray right) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Nd4j.valueArrayOf(right.shape(), left.getDouble(0)).eq(right);
            } else {
                result = left.eq(right.getDouble(0));
            }
            return result.reshape(resultShape);
        }
        return left.eq(right);
    }

    public static INDArray atan2(INDArray left, INDArray right, DataBuffer.Type bufferType) {
        if (left.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(left.shape(), right.shape());
            INDArray result;
            if (left.length() == 1) {
                result = Transforms.atan2(Nd4j.valueArrayOf(right.shape(), left.getDouble(0)), right);
            } else {
                result = Transforms.atan2(left, valueArrayOf(resultShape, right.getDouble(0), bufferType));
            }
            return result.reshape(resultShape);
        }
        return Transforms.atan2(left, right);
    }

    public static INDArray getGreaterThanMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        if (mask.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(mask.shape(), right.shape());
            if (mask.length() == 1) {
                mask = Nd4j.valueArrayOf(right.shape(), mask.getDouble(0));
                Nd4j.getExecutioner().exec(
                    new OldGreaterThan(mask, right, mask, mask.length())
                );
            } else {
                Nd4j.getExecutioner().exec(
                    new OldGreaterThan(mask,
                        valueArrayOf(mask.shape(), right.getDouble(0), bufferType),
                        mask,
                        mask.length()
                    )
                );
            }
            return mask.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(
            new OldGreaterThan(mask, right, mask, mask.length())
        );
        return mask;
    }

    public static INDArray getGreaterThanOrEqualToMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        if (mask.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(mask.shape(), right.shape());
            if (mask.length() == 1) {
                mask = Nd4j.valueArrayOf(right.shape(), mask.getDouble(0));
                Nd4j.getExecutioner().exec(
                    new OldGreaterThanOrEqual(mask, right, mask, mask.length())
                );
            } else {
                Nd4j.getExecutioner().exec(
                    new OldGreaterThanOrEqual(mask,
                        valueArrayOf(mask.shape(), right.getDouble(0), bufferType),
                        mask,
                        mask.length()
                    )
                );
            }
            return mask.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(
            new OldGreaterThanOrEqual(mask, right, mask, mask.length())
        );
        return mask;
    }

    public static INDArray getLessThanMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        if (mask.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(mask.shape(), right.shape());
            if (mask.length() == 1) {
                mask = Nd4j.valueArrayOf(right.shape(), mask.getDouble(0));
                Nd4j.getExecutioner().exec(
                    new OldLessThan(mask, right, mask, mask.length())
                );
            } else {
                Nd4j.getExecutioner().exec(
                    new OldLessThan(mask,
                        valueArrayOf(mask.shape(), right.getDouble(0), bufferType),
                        mask,
                        mask.length()
                    )
                );
            }
            return mask.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(
            new OldLessThan(mask, right, mask, mask.length())
        );
        return mask;
    }

    public static INDArray getLessThanOrEqualToMask(INDArray mask, INDArray right, DataBuffer.Type bufferType) {
        if (mask.length() == 1 || right.length() == 1) {
            long[] resultShape = Shape.broadcastOutputShape(mask.shape(), right.shape());
            if (mask.length() == 1) {
                mask = Nd4j.valueArrayOf(right.shape(), mask.getDouble(0));
                Nd4j.getExecutioner().exec(
                    new OldLessThanOrEqual(mask, right, mask, mask.length())
                );
            } else {
                Nd4j.getExecutioner().exec(
                    new OldLessThanOrEqual(mask,
                        valueArrayOf(mask.shape(), right.getDouble(0), bufferType),
                        mask,
                        mask.length()
                    )
                );
            }
            return mask.reshape(resultShape);
        }
        Nd4j.getExecutioner().exec(
            new OldLessThanOrEqual(mask, right, mask, mask.length())
        );
        return mask;
    }

    private static boolean shapeAIsSmallerThanShapeB(long[] shapeA, long[] shapeB) {
        if (shapeA.length != shapeB.length) {
            return shapeA.length < shapeB.length;
        }
        for (int ind = 0; ind < shapeA.length; ind++) {
            if (shapeA[ind] < shapeB[ind]) {
                return true;
            }
        }
        return false;
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

                //dup is required before reshaping due to a bug in ND4J that doesn't always correctly
                //duplicate true vectors.
                return result.dup().reshape(newShape);
            } else {
                return result;
            }
        }

    }
}
