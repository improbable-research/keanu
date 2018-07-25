package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertArrayEquals;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class DoubleBinaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_tensor_THEN_transform() {

        UniformVertex matrix = new DistributionVertexBuilder()
            .shaped(2, 2)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 5.)
            .uniform();
        matrix.setAndCascade(DoubleTensor.create(2.5, new int[]{2, 2}));
        UniformVertex matrixB = new DistributionVertexBuilder()
            .shaped(2, 2)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 5.)
            .uniform();
        matrixB.setAndCascade(DoubleTensor.create(3.5, new int[]{2, 2}));

        DoubleVertex matrixLambda = new DoubleBinaryOpLambda<>(
            matrix.getShape(), matrix, matrixB,
            (val, valB) -> val.plus(valB)
        );

        assertArrayEquals(new double[]{6, 6, 6, 6}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
    }

}
