package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class DoubleUnaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new DistributionVertexBuilder()
            .shaped(2, 2)
            .withInput(ParameterName.MIN, 0.0)
            .withInput(ParameterName.MAX, 5.0)
            .uniform();
        matrix.setAndCascade(DoubleTensor.create(2.5, new int[] {2, 2}));
        DoubleVertex matrixLambda = new DoubleUnaryOpLambda<>(matrix.getShape(), matrix, (val) -> val.times(2));

        assertArrayEquals(new double[]{5, 5, 5, 5}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
