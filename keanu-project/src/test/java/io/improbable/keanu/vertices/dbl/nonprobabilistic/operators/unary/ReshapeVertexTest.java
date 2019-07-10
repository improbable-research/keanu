package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class ReshapeVertexTest {

    @Test
    public void reshapeVertex() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex reshapeVertex = a.reshape(4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new long[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, reshapeVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void reshapeCorrectlyReshapesPartialDerivative() {
        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        MatrixMultiplicationVertex N = (MatrixMultiplicationVertex) m.matrixMultiply(alpha);

        DoubleVertex reshapedN = N.reshape(4, 1);

        DoubleTensor dReshapedNWrtmForward = Differentiator.forwardModeAutoDiff(m, reshapedN).of(reshapedN);
        DoubleTensor dReshapedNWrtmBackward = Differentiator.reverseModeAutoDiff(reshapedN, ImmutableSet.of(m, alpha)).withRespectTo(m);

        Assert.assertArrayEquals(new long[]{4, 1, 2, 2}, dReshapedNWrtmForward.getShape());
        Assert.assertArrayEquals(new long[]{4, 1, 2, 2}, dReshapedNWrtmBackward.getShape());

        Assert.assertArrayEquals(dReshapedNWrtmBackward.asFlatDoubleArray(), dReshapedNWrtmForward.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void flatPartialDerivativeIsTheSameAfterReshape() {
        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        MatrixMultiplicationVertex N = (MatrixMultiplicationVertex) m.matrixMultiply(a);

        DoubleTensor dNdm = Differentiator.reverseModeAutoDiff(N, m).withRespectTo(m);
        DoubleTensor dNda = Differentiator.reverseModeAutoDiff(N, a).withRespectTo(a);

        double[] nWrtMpartialsBeforeReshape = dNdm.asFlatDoubleArray();
        double[] nWrtApartialsBeforeReshape = dNda.asFlatDoubleArray();

        DoubleVertex reshapedN = N.reshape(4, 1);
        DoubleTensor reshapedPartialWrtM = Differentiator.reverseModeAutoDiff(reshapedN, m).withRespectTo(m);
        DoubleTensor reshapedPartialWrtA = Differentiator.reverseModeAutoDiff(reshapedN, a).withRespectTo(a);

        Assert.assertArrayEquals(nWrtMpartialsBeforeReshape, reshapedPartialWrtM.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(nWrtApartialsBeforeReshape, reshapedPartialWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void partialCorrectlyFlowsThroughReshape() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex C = A.plus(B);

        DoubleVertex D = C.reshape(4, 1);

        UniformVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 4, 1));

        MultiplicationVertex F = D.times(E);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, F).of(F);
        PartialsOf backwardA = Differentiator.reverseModeAutoDiff(F, ImmutableSet.of(A));

        DoubleTensor forwardWrtB = Differentiator.forwardModeAutoDiff(B, F).of(F);
        PartialsOf backwardB = Differentiator.reverseModeAutoDiff(F, ImmutableSet.of(B));

        Assert.assertArrayEquals(new long[]{4, 1, 2, 2}, forwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backwardA.withRespectTo(A).asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(new long[]{4, 1, 2, 2}, forwardWrtB.getShape());
        Assert.assertArrayEquals(forwardWrtB.asFlatDoubleArray(), backwardB.withRespectTo(B).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void partialCorrectlyFlowsThroughTwoReshapes() {
        UniformVertex A = new UniformVertex(new long[]{2, 2, 2, 2}, 0, 10);
        A.setValue(A.sample());

        UniformVertex B = new UniformVertex(new long[]{2, 2, 2, 2}, 0, 10);
        B.setValue(B.sample());

        DoubleVertex C = A.plus(B);

        DoubleVertex D = C.reshape(4, 2, 2);
        DoubleVertex E = D.reshape(4, 4);

        DoubleTensor forwardWrtA = Differentiator.forwardModeAutoDiff(A, E).of(E);
        PartialsOf backward = Differentiator.reverseModeAutoDiff(E, ImmutableSet.of(A, B));

        Assert.assertArrayEquals(new long[]{4, 4, 2, 2, 2, 2}, forwardWrtA.getShape());
        Assert.assertArrayEquals(forwardWrtA.asFlatDoubleArray(), backward.withRespectTo(A).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{4, 4}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(1.5).reshape(2, 2, 2, 2);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-10, 1e-10);
    }

}
