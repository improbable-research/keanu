package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class DoubleBinaryOpLambdaTest {

    class MatrixLambda extends DoubleBinaryOpVertex {

        public MatrixLambda(DoubleVertex left, DoubleVertex right) {
            super(left, right);
        }

        @Override
        protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
            return l.plus(r);
        }

        @Override
        protected DualNumber dualOp(DualNumber l, DualNumber r) {
            return null;
        }
    }
    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new UniformVertex(new int[]{2, 2}, 0, 5);
        matrix.setAndCascade(DoubleTensor.create(2.5, new int[]{2, 2}));
        UniformVertex matrixB = new UniformVertex(new int[]{2, 2}, 0, 5);
        matrixB.setAndCascade(DoubleTensor.create(3.5, new int[]{2, 2}));

        DoubleVertex matrixLambda = new MatrixLambda(matrix, matrixB);

        assertArrayEquals(new double[]{6, 6, 6, 6}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
