package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;

import java.util.ArrayList;
import java.util.List;

public class TensorMulByMatrixMul {

    public static DoubleTensor tensorMmul(DoubleTensor a, DoubleTensor b, int[][] axes) {

        long[] aShape = a.getShape();
        long[] bShape = b.getShape();

        int validationLength = Math.min(axes[0].length, axes[1].length);
        for (int i = 0; i < validationLength; i++) {
            if (aShape[axes[0][i]] != bShape[axes[1][i]])
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            if (axes[0][i] < 0)
                axes[0][i] += aShape.length;
            if (axes[1][i] < 0)
                axes[1][i] += bShape.length;

        }

        List<Integer> listA = new ArrayList<>();
        for (int i = 0; i < aShape.length; i++) {
            if (!Ints.contains(axes[0], i))
                listA.add(i);
        }

        int[] newAxesA = Ints.concat(Ints.toArray(listA), axes[0]);

        List<Integer> listB = new ArrayList<>();
        for (int i = 0; i < bShape.length; i++) {
            if (!Ints.contains(axes[1], i))
                listB.add(i);
        }

        int[] newAxesB = Ints.concat(axes[1], Ints.toArray(listB));

        int n2 = 1;
        int aLength = Math.min(aShape.length, axes[0].length);
        for (int i = 0; i < aLength; i++) {
            n2 *= aShape[axes[0][i]];
        }

        //if listA and listB are empty these do not initialize.
        //so initializing with {1} which will then get overridden if not empty
        long[] newShapeA = {-1, n2};
        long[] oldShapeA;
        if (listA.size() == 0) {
            oldShapeA = new long[0];
        } else {
            oldShapeA = Longs.toArray(listA);
            for (int i = 0; i < oldShapeA.length; i++) {
                oldShapeA[i] = aShape[Ints.checkedCast(oldShapeA[i])];
            }
        }

        int n3 = 1;
        int bNax = Math.min(bShape.length, axes[1].length);
        for (int i = 0; i < bNax; i++) {
            n3 *= bShape[axes[1][i]];
        }

        long[] newShapeB = {n3, -1};
        long[] oldShapeB;
        if (listB.size() == 0) {
            oldShapeB = new long[0];
        } else {
            oldShapeB = Longs.toArray(listB);
            for (int i = 0; i < oldShapeB.length; i++) {
                oldShapeB[i] = bShape[Ints.checkedCast(oldShapeB[i])];
            }
        }

        DoubleTensor at = a.permute(newAxesA).reshape(newShapeA);
        DoubleTensor bt = b.permute(newAxesB).reshape(newShapeB);
        DoubleTensor ret = at.matrixMultiply(bt);

        long[] aPlusB = Longs.concat(oldShapeA, oldShapeB);
        return ret.reshape(aPlusB);
    }
}
