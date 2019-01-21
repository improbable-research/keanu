package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
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
        a.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 6).reshape(1, 2, 3));

        DoubleVertex c = a.times(b);

        PermuteVertex permute = new PermuteVertex(c, 2, 0, 1);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(a, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, a).withRespectTo(a);

        System.out.println(Arrays.toString(forwardWrtA.getShape()));
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPermuteRankFourVertex() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 16).reshape(2, 1, 4, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 16).reshape(2, 1, 4, 2));

        DoubleVertex c = a.times(b);

        PermuteVertex permute = new PermuteVertex(c, 3, 2, 0, 1);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(a, permute).of(permute);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(permute, a).withRespectTo(a);

        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPartialCorrectlyFlowThroughRankThreePermute() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 8).reshape(4, 1, 2));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 8).reshape(4, 1, 2));

        DoubleVertex C = A.plus(B);

        DoubleVertex D = C.permute(2, 0, 1);

        UniformVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.arange(0, 8).reshape(2, 4, 1));

        MultiplicationVertex F = D.times(E);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, F).of(F);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(F, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(new long[]{2, 4, 1, 4, 1, 2}, forwardWrtA.getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 1, 4, 1, 2}, backwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPartialCorrectlyFlowThroughRankTwoPermute() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        MultiplicationVertex C = A.times(B);

        PermuteVertex D = C.permute(1, 0);

        UniformVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        MultiplicationVertex F = D.times(E);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, F).of(F);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(F, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(new long[]{2, 2, 2, 2}, forwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPartialCorrectlyFlowThroughReversedPermute() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        MultiplicationVertex C = A.times(B);

        PermuteVertex D = C.permute(2, 0, 1);
        PermuteVertex E = D.permute(1, 2, 0);

        Assert.assertArrayEquals(C.getValue().asFlatDoubleArray(), E.getValue().asFlatDoubleArray(), 1e-6);

        DoubleTensor reversedForwardWrtA = Differentiator.forwardModeAutoDiff(A, E).of(E);
        DoubleTensor reversedBackwardWrtA = Differentiator.reverseModeAutoDiff(E, ImmutableSet.of(A)).withRespectTo(A);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, C).of(C);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), reversedForwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedForwardWrtA.asFlatDoubleArray(), forwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPartialCorrectlyFlowThroughRankTwoReversedPermute() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        MultiplicationVertex C = A.times(B);

        PermuteVertex D = C.permute(1, 0);
        PermuteVertex E = D.permute(1, 0);

        Assert.assertArrayEquals(C.getValue().asFlatDoubleArray(), E.getValue().asFlatDoubleArray(), 1e-6);

        DoubleTensor reversedForwardWrtA = Differentiator.forwardModeAutoDiff(A, E).of(E);
        DoubleTensor reversedBackwardWrtA = Differentiator.reverseModeAutoDiff(E, ImmutableSet.of(A)).withRespectTo(A);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, C).of(C);
        DoubleTensor backwardWrtA = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A)).withRespectTo(A);

        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), reversedForwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedForwardWrtA.asFlatDoubleArray(), forwardWrtA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(reversedBackwardWrtA.asFlatDoubleArray(), backwardWrtA.asFlatDoubleArray(), 1e-6);
    }


    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{4, 4, 4}, -10.0, 10.0);
        PermuteVertex outputVertex = inputVertex.permute(1, 2, 0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-10, 1e-10);
    }
}
