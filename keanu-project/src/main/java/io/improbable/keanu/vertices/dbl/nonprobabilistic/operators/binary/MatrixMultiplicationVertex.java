package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.Arrays;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex implements Differentiable {

    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param a vertex A
     * @param b vertex B
     */
    public MatrixMultiplicationVertex(DoubleVertex a, DoubleVertex b) {
        super(getResultingShape(a.getShape(), b.getShape()), a, b);
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        DualNumber aDual = dualNumbers.get(a);
        DualNumber bDual = dualNumbers.get(b);
        return aDual.matrixMultiplyBy(bDual);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.matrixMultiply(b);
    }

    private static int[] getResultingShape(int[] left, int[] right) {
        if (left.length != 2 || right.length != 2) {
            throw new IllegalArgumentException("Matrix multiply must be used on matrices");
        }

        if (left[1] != right[0]) {
            throw new IllegalArgumentException("Can not multiply matrices of shapes " + Arrays.toString(left) + " X " + Arrays.toString(right));
        }

        return new int[]{left[0], right[1]};
    }
}
