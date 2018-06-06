package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class DoubleBinaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new UniformVertex(new int[]{2, 2}, 0, 5);
        matrix.setAndCascade(2.5);
        UniformVertex matrixB = new UniformVertex(new int[]{2, 2}, 0, 5);
        matrixB.setAndCascade(3.5);

        DoubleVertex matrixLambda = new DoubleBinaryOpLambda<>(
            matrix.getShape(), matrix, matrixB,
            (val, valB) -> val.plus(valB)
        );

        assertArrayEquals(new double[]{6, 6, 6, 6}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
