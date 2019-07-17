package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.testWithFiniteDifference;
import static org.hamcrest.MatcherAssert.assertThat;

public class PermuteVertexTest {

    @Test
    public void canPermuteForTranpose() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));

        DoubleVertex transpose = a.permute(1, 0);

        Assert.assertArrayEquals(new long[]{3, 2}, transpose.getShape());
        Assert.assertArrayEquals(a.getValue().transpose().asFlatDoubleArray(), transpose.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void changesMatchGradientRankTwo() {
        testWithFiniteDifference(v -> v.permute(1, 0), new long[]{4, 5});
        testWithFiniteDifference(v -> v.permute(0, 1), new long[]{4, 5});
    }

    @Test
    public void changesMatchGradientRankThree() {
        testWithFiniteDifference(v -> v.permute(1, 2, 0), new long[]{4, 3, 2});
        testWithFiniteDifference(v -> v.permute(0, 1, 2), new long[]{4, 3, 2});
        testWithFiniteDifference(v -> v.permute(2, 1, 0), new long[]{4, 3, 2});
    }

    @Test
    public void canCalculatePermutedShapeOfAutoDiffRankThree() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        DoubleVertex c = a.times(b);

        DoubleVertex permute = c.permute(2, 0, 1);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(a, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, a).withRespectTo(a);

        Assert.assertArrayEquals(new long[]{3, 1, 2, 1, 2, 3}, backwardWrtA.getShape());
        Assert.assertArrayEquals(new long[]{3, 1, 2, 1, 2, 3}, forwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canCalculatePermutedShapeOfAutoDiffRankFour() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 16).reshape(2, 1, 4, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 16).reshape(2, 1, 4, 2));

        DoubleVertex c = a.times(b);

        DoubleVertex permute = c.permute(3, 2, 0, 1);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(a, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, a).withRespectTo(a);

        Assert.assertArrayEquals(new long[]{2, 4, 2, 1, 2, 1, 4, 2}, backwardWrtA.getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 1, 2, 1, 4, 2}, forwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canCalculateCorrectPermutedShapeWithUpstreamSum() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        DoubleVertex C = A.plus(B);

        DoubleVertex sum = C.sum(2);

        DoubleVertex permute = sum.permute(1, 0);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(new long[]{2, 1, 1, 2, 3}, forwardWrtA.getShape());
        Assert.assertArrayEquals(new long[]{2, 1, 1, 2, 3}, backwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canCalculateCorrectPermutedShapeWithDownstreamSum() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        DoubleVertex C = A.plus(B);

        DoubleVertex permute = C.permute(0, 2, 1);

        DoubleVertex sum = permute.sum(0);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, sum).of(sum);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(sum, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(new long[]{3, 2, 1, 2, 3}, forwardWrtA.getShape());
        Assert.assertArrayEquals(new long[]{3, 2, 1, 2, 3}, backwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void autoDiffIsUnchangedWhenPermuteIsReversed() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        DoubleVertex C = A.times(B);

        DoubleVertex permute = C.permute(2, 0, 1);
        DoubleVertex revertThePermute = permute.permute(1, 2, 0);

        assertThat(C.getValue(), valuesAndShapesMatch(revertThePermute.getValue()));

        DoubleTensor reversedForwardWrtA = Differentiator.forwardModeAutoDiff(A, revertThePermute).of(revertThePermute);
        DoubleTensor reversedBackwardWrtA = Differentiator.reverseModeAutoDiff(revertThePermute, ImmutableSet.of(A)).withRespectTo(A);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, C).of(C);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), reversedForwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedForwardWrtA.asFlatDoubleArray(), forwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

}
