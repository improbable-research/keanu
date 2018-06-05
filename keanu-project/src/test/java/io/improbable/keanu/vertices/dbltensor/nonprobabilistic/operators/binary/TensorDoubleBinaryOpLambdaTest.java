package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class TensorDoubleBinaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        TensorUniformVertex matrix = new TensorUniformVertex(new int[]{2, 2}, 0, 5);
        matrix.setAndCascade(2.5);
        TensorUniformVertex matrixB = new TensorUniformVertex(new int[]{2, 2}, 0, 5);
        matrixB.setAndCascade(3.5);

        DoubleTensorVertex matrixLambda = new TensorDoubleBinaryOpLambda<>(matrix, matrixB, (val, valB) -> val.plus(valB));

        assertArrayEquals(new double[]{6, 6, 6, 6}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
