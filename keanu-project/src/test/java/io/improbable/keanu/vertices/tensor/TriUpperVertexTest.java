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

public class TriUpperVertexTest {

    @Test
    public void doesSimpleTriUpper() {
        DoubleVertex input = ConstantVertex.of(1., 2., 3., 4.).reshape(2, 2);
        DoubleVertex result = input.triUpper(0);

        assertThat(result.getValue(), valuesAndShapesMatch(DoubleTensor.create(1, 2, 0, 4).reshape(2, 2)));
    }

    @Test
    public void changesMatchGradientForward() {
        assertChangesMatchGradientForward(new long[]{2, 2}, DoubleTensor.arange(1, 3), 0);
        assertChangesMatchGradientForward(new long[]{2, 2}, DoubleTensor.arange(1, 3), 1);
    }

    @Test
    public void changesMatchGradientReverse() {
        assertChangesMatchGradientReverse(new long[]{2, 2}, DoubleTensor.arange(1, 3), 0);
        assertChangesMatchGradientReverse(new long[]{2, 2}, DoubleTensor.arange(1, 3), 1);
    }

    @Test
    public void changesMatchGradientWithBatchTriUpperForward() {
        assertChangesMatchGradientForward(new long[]{2, 4, 4}, DoubleTensor.arange(1, 17).reshape(4, 4), 0);
        assertChangesMatchGradientForward(new long[]{2, 4, 4}, DoubleTensor.arange(1, 17).reshape(4, 4), 1);
    }

    @Test
    public void changesMatchGradientWithBatchTriUpperReverse() {
        assertChangesMatchGradientReverse(new long[]{2, 4, 4}, DoubleTensor.arange(1, 17).reshape(4, 4), 0);
        assertChangesMatchGradientReverse(new long[]{2, 4, 4}, DoubleTensor.arange(1, 17).reshape(4, 4), 1);
    }

    @Test
    public void changesMatchGradientWithHighDimensionBatchTriUpperForward() {
        assertChangesMatchGradientForward(new long[]{3, 2, 4, 4}, DoubleTensor.arange(1, 33).reshape(2, 4, 4), 0);
        assertChangesMatchGradientForward(new long[]{3, 2, 4, 4}, DoubleTensor.arange(1, 33).reshape(2, 4, 4), 1);
    }

    @Test
    public void changesMatchGradientWithHighDimensionBatchTriUpperReverse() {
        assertChangesMatchGradientReverse(new long[]{3, 2, 4, 4}, DoubleTensor.arange(1, 33).reshape(2, 4, 4), 0);
        assertChangesMatchGradientReverse(new long[]{3, 2, 4, 4}, DoubleTensor.arange(1, 33).reshape(2, 4, 4), 1);
    }

    private void assertChangesMatchGradientForward(long[] inputShape, DoubleTensor postOpFactor, int k) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.triUpper(k);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    private void assertChangesMatchGradientReverse(long[] inputShape, DoubleTensor postOpFactor, int k) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex opResult = inputA.triUpper(k);
        DoubleVertex outputVertex = opResult.times(
            new ConstantDoubleVertex(postOpFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}
