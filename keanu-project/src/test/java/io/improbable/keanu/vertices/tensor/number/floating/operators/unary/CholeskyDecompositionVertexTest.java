package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class CholeskyDecompositionVertexTest {

    @Test
    public void canFindCholeskyDecomposition() {
        //Example from: https://en.wikipedia.org/wiki/Cholesky_decomposition
        DoubleVertex A = new ConstantDoubleVertex(new double[]{
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        }).reshape(3, 3);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            2, 0, 0,
            6, 1, 0,
            -8, 5, 3
        }, 3, 3);

        DoubleVertex actual = A.choleskyDecomposition();

        assertThat(actual.getValue(), valuesWithinEpsilonAndShapesMatch(expected, 1e-10));
    }
}
