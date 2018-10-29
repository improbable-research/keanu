package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class DoubleUnaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new UniformVertex(new long[]{2, 2}, 0, 5);
        matrix.setAndCascade(DoubleTensor.create(2.5, new long[]{2, 2}));
        DoubleVertex matrixLambda = matrix.lambda((val) -> val.times(2), null, null);

        assertArrayEquals(new double[]{5, 5, 5, 5}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
