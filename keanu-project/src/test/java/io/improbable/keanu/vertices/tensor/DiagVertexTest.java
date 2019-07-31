package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;

public class DiagVertexTest {

    @Test
    @Ignore
    public void changesMatchGradientForward() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 2}, -10.0, 10.0);
        DoubleVertex diag = inputA.diag();
        DoubleVertex outputVertex = diag.times(
            new ConstantDoubleVertex(DoubleTensor.arange(1, 3))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    @Test
    public void changesMatchGradientReverse() {
        assertChangesMatchesReverseGradient(new long[]{2, 2}, DoubleTensor.arange(1, 3));
    }

    @Test
    public void changesMatchGradientReverseWithWideMatrix() {
        assertChangesMatchesReverseGradient(new long[]{2, 3}, DoubleTensor.arange(1, 2));
    }

    @Test
    public void changesMatchGradientReverseWithBatchNarrowMatrix() {
        assertChangesMatchesReverseGradient(new long[]{3, 2}, DoubleTensor.arange(1, 2));
    }

    @Test
    public void changesMatchGradientReverseWithBatchRank3() {
        assertChangesMatchesReverseGradient(new long[]{2, 3, 2}, DoubleTensor.arange(1, 2));
    }

    @Test(expected = IllegalArgumentException.class)
    public void changesMatchGradientReverseWithScalar() {
        assertChangesMatchesReverseGradient(new long[0], DoubleTensor.scalar(2));
    }

    public void assertChangesMatchesReverseGradient(long[] inputShape, DoubleTensor postDiagFactor) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex diag = inputA.diag();
        DoubleVertex outputVertex = diag.times(
            new ConstantDoubleVertex(postDiagFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

}
