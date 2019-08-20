package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;
import static org.hamcrest.MatcherAssert.assertThat;

public class CholeskyDecompositionVertexTest {

    @Test
    public void canFindCholeskyDecomposition() {
        //Example from: https://en.wikipedia.org/wiki/Cholesky_decomposition
        DoubleVertex A = new ConstantDoubleVertex(
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        ).reshape(3, 3);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            2, 0, 0,
            6, 1, 0,
            -8, 5, 3
        }, 3, 3);

        DoubleVertex actual = A.choleskyDecomposition();

        assertThat(actual.getValue(), valuesWithinEpsilonAndShapesMatch(expected, 1e-10));
    }

    @Test
    public void differenceMatchesGradientForward() {
        UniformVertex inputVertex = new UniformVertex(new long[]{3, 3}, 1.0, 25.0);
        DoubleVertex L = inputVertex.choleskyDecomposition();

        finiteDifferenceMatchesForwardModeGradient(
            ImmutableList.of(inputVertex), L, 0.001, 1e-5);
    }

    @Test
    public void differenceMatchesGradientReverse() {
        UniformVertex inputVertex = new UniformVertex(new long[]{3, 3}, 1.0, 25.0);
        inputVertex.setValue(DoubleTensor.create(
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        ).reshape(3, 3));

        //inputVertex.getValue().choleskyDecomposition().minus(inputVertex.getValue().plus(DoubleTensor.create(0.001,0,0,0,0,0,0,0,0).reshape(3,3)).choleskyDecomposition())
        DoubleVertex L = inputVertex.choleskyDecomposition();

        finiteDifferenceMatchesReverseModeGradient(
            ImmutableList.of(inputVertex), L, 0.001, 1e-5);
    }
}
