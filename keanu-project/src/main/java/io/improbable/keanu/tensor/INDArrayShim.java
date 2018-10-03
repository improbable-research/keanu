package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class provides shim methods for the ND4J INDArray class. The INDArray broadcast operations
 * are currently broken (https://github.com/deeplearning4j/deeplearning4j/issues/5893). Until this
 * is fixed in the ND4J codebase, these methods can be used to work around the issue. The need for
 * this should be reevaluated each time the ND4J dependency is updated.
 *
 * <p>To work around another issue in ND4J where you cannot broadcast a higher rank tensor onto a
 * lower rank tensor, the shim broadcast operations ensure the higher rank tensor is always being
 * operated on. In the case of subtract and minus, this requires a small change in the logic, as A -
 * B != B - A and A / B != B / A.
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
      return execBroadcast(a, b, new BroadcastMulOp(a, b, a.dup(), broadcastDimensions));
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
      return broadcastMultiply(b, a.rdiv(1.0));
    } else {
      int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
      return execBroadcast(a, b, new BroadcastDivOp(a, b, a.dup(), broadcastDimensions));
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
      return execBroadcast(a, b, new BroadcastAddOp(a, b, a.dup(), broadcastDimensions));
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
      return broadcastPlus(a.neg(), b);
    } else {
      int[] broadcastDimensions = getBroadcastDimensions(a.shape(), b.shape());
      return execBroadcast(a, b, new BroadcastSubOp(a, b, a.dup(), broadcastDimensions));
    }
  }

  private static INDArray execBroadcast(INDArray a, INDArray b, BroadcastOp op) {
    int[] executeAlong = getBroadcastAlongDimensions(a.shape(), b.shape());
    return Nd4j.getExecutioner().exec(op, executeAlong);
  }

  private static int[] getBroadcastDimensions(int[] shapeA, int[] shapeB) {
    int maxRank = Math.max(shapeA.length, shapeB.length);

    if (shapeA.length < shapeB.length) {
      shapeA = TensorShape.shapeToDesiredRankByPrependingOnes(shapeA, shapeB.length);
    } else {
      shapeB = TensorShape.shapeToDesiredRankByPrependingOnes(shapeB, shapeA.length);
    }

    List<Integer> along = new ArrayList<>();

    for (int i = maxRank - 1; i >= 0; i--) {
      if (shapeA[i] != shapeB[i]) {
        along.add(i);
      }
    }
    return Ints.toArray(along);
  }

  private static int[] getBroadcastAlongDimensions(int[] shapeA, int[] shapeB) {
    int maxRank = Math.max(shapeA.length, shapeB.length);

    if (shapeA.length < shapeB.length) {
      shapeA = TensorShape.shapeToDesiredRankByPrependingOnes(shapeA, shapeB.length);
    } else {
      shapeB = TensorShape.shapeToDesiredRankByPrependingOnes(shapeB, shapeA.length);
    }

    List<Integer> along = new ArrayList<>();

    for (int i = maxRank - 1; i >= 0; i--) {
      if (shapeA[i] == shapeB[i]) {
        along.add(i);
      }
    }
    return Ints.toArray(along);
  }
}
