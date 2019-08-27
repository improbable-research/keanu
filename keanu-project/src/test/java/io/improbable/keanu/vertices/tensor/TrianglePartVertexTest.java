package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;
import static org.hamcrest.MatcherAssert.assertThat;

public class TrianglePartVertexTest {

    @Test
    public void doesSimpleUpper() {
        DoubleVertex input = ConstantVertex.of(1., 2., 3., 4, 5, 6, 7, 8, 9).reshape(3, 3);
        DoubleVertex result = input.trianglePart(true);

        assertThat(result.getValue(), valuesAndShapesMatch(DoubleTensor.create(1, 2, 3, 5, 6, 9)));
    }

    @Test
    public void changesMatchGradientForward() {
        assertChangesMatchGradientForward(new long[]{3, 3}, DoubleTensor.arange(1, 7), false);
        assertChangesMatchGradientForward(new long[]{3, 3}, DoubleTensor.arange(1, 7), true);
    }

    @Test
    public void changesMatchGradientReverse() {
        assertChangesMatchGradientReverse(new long[]{3, 3}, DoubleTensor.arange(1, 7), false);
        assertChangesMatchGradientReverse(new long[]{3, 3}, DoubleTensor.arange(1, 7), true);
    }

    private void assertChangesMatchGradientForward(long[] inputShape, DoubleTensor postOpFactor, boolean upperPart) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.trianglePart(upperPart);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    private void assertChangesMatchGradientReverse(long[] inputShape, DoubleTensor postOpFactor, boolean upperPart) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.trianglePart(upperPart);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}
