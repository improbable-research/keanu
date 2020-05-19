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

public class FillTriangularVertexTest {

    @Test
    public void doesSimpleUpperAndLower() {
        DoubleVertex input = ConstantVertex.of(1., 2., 3.);
        DoubleVertex result = input.fillTriangular(true, true);

        assertThat(result.getValue(), valuesAndShapesMatch(DoubleTensor.create(1, 2, 2, 3).reshape(2, 2)));
    }

    @Test
    public void changesMatchGradientForward() {
        assertChangesMatchGradientForward(new long[]{6}, DoubleTensor.arange(1, 4), false, false);
        assertChangesMatchGradientForward(new long[]{6}, DoubleTensor.arange(1, 4), true, false);
        assertChangesMatchGradientForward(new long[]{6}, DoubleTensor.arange(1, 4), false, true);
        assertChangesMatchGradientForward(new long[]{6}, DoubleTensor.arange(1, 4), true, true);
    }

    @Test
    public void changesMatchGradientReverse() {
        assertChangesMatchGradientReverse(new long[]{6}, DoubleTensor.arange(1, 4), false, false);
        assertChangesMatchGradientReverse(new long[]{6}, DoubleTensor.arange(1, 4), true, false);
        assertChangesMatchGradientReverse(new long[]{6}, DoubleTensor.arange(1, 4), false, true);
        assertChangesMatchGradientReverse(new long[]{6}, DoubleTensor.arange(1, 4), true, true);
    }

    private void assertChangesMatchGradientForward(long[] inputShape, DoubleTensor postOpFactor, boolean fillUpper, boolean fillLower) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.fillTriangular(fillUpper, fillLower);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    private void assertChangesMatchGradientReverse(long[] inputShape, DoubleTensor postOpFactor, boolean fillUpper, boolean fillLower) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.fillTriangular(fillUpper, fillLower);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}
