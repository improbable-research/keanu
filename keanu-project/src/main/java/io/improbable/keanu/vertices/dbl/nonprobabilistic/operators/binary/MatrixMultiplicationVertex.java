package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.Arrays;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class MatrixMultiplicationVertex extends DoubleBinaryOpVertex implements Differentiable {

    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param a vertex A
     * @param b vertex B
     */
    public MatrixMultiplicationVertex(DoubleVertex a, DoubleVertex b) {
        super(getResultingShape(a.getShape(), b.getShape()), a, b, (a_,b_) -> a_.matrixMultiply(b_), (a_,b_) -> a_.matrixMultiplyBy(b_));
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
