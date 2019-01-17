package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class PermuteVertexTest {

    @Test
    public void canPermuteForTranpose() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        PermuteVertex permute = new PermuteVertex(a, 1, 0);
        permute.getValue();

        Assert.assertArrayEquals(new long[]{2, 2}, permute.getShape());
        Assert.assertArrayEquals(a.getValue().transpose().asFlatDoubleArray(), permute.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPermuteRankThreeVertex() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        PermuteVertex permute = new PermuteVertex(a, 2, 0, 1);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(a, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, a).withRespectTo(a);

        System.out.println(forwardWrtA);
        System.out.println(backwardWrtA);

    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{4, 4}, -10.0, 10.0);
        PermuteVertex outputVertex = inputVertex.permute(1, 0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-10, 1e-10);
    }
}
