package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class PermuteVertexTest {

    @Test
    public void canPermuteForTranpose() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        PermuteVertex PermuteVertex = new PermuteVertex(a, 1, 0);
        PermuteVertex.getValue();

        Assert.assertArrayEquals(new long[]{2, 2}, PermuteVertex.getShape());
        Assert.assertArrayEquals(a.getValue().transpose().asFlatDoubleArray(), PermuteVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{4, 4}, -10.0, 10.0);
        PermuteVertex outputVertex = inputVertex.times(1.5).permute(1, 0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-10, 1e-10);
    }
}
