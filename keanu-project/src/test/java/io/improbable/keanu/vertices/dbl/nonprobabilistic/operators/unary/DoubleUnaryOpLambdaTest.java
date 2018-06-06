package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class DoubleUnaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new UniformVertex(new int[]{2, 2}, 0, 5);
        matrix.setAndCascade(2.5);
        DoubleVertex matrixLambda = new DoubleUnaryOpLambda<>(matrix.getShape(), matrix, (val) -> val.times(2));

        assertArrayEquals(new double[]{5, 5, 5, 5}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
