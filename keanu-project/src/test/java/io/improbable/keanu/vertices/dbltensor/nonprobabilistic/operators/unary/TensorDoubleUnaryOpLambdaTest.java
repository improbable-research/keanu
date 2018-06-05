package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class TensorDoubleUnaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        TensorUniformVertex matrix = new TensorUniformVertex(new int[]{2, 2}, 0, 5);
        matrix.setAndCascade(2.5);
        DoubleTensorVertex matrixLambda = new TensorDoubleUnaryOpLambda<>(matrix, (val) -> val.times(2));

        assertArrayEquals(new double[]{5, 5, 5, 5}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
