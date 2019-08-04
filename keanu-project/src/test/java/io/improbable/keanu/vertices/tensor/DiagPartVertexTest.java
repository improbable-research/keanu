package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class DiagPartVertexTest {

    @Test
    public void changesMatchGradient() {
        assertChangesMatchGradient(new long[]{2, 2}, DoubleTensor.arange(1, 3));
    }

    @Test
    public void changesMatchGradientWithBatchDiag() {
        assertChangesMatchGradient(new long[]{2, 4, 4}, DoubleTensor.arange(1, 9).reshape(2, 4));
    }

    @Test
    public void changesMatchGradientWithHighDimensionBatchDiag() {
        assertChangesMatchGradient(new long[]{3, 2, 4, 4}, DoubleTensor.arange(1, 25).reshape(3, 2, 4));
    }

    private void assertChangesMatchGradient(long[] inputShape, DoubleTensor postDiagFactor) {
        UniformVertex inputA = new UniformVertex(inputShape, -10.0, 10.0);
        DoubleVertex diag = inputA.diagPart();
        DoubleVertex outputVertex = diag.times(
            new ConstantDoubleVertex(postDiagFactor)
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfInputIsRank1() {
        UniformVertex inputA = new UniformVertex(new long[]{2}, -10.0, 10.0);
        inputA.diagPart();
    }
}
