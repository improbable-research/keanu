package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ReshapeVertexTest {

    @Test
    public void reshapeVertex() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        ReshapeVertex reshapeVertex = new ReshapeVertex(a, 4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, reshapeVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void reshapeCorrectlyReshapesPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        ReshapeVertex reshapedN = new ReshapeVertex(N, 4, 1);

        PartialDerivatives forward = reshapedN.getDualNumber().getPartialDerivatives();
        PartialDerivatives backward = Differentiator.reverseModeAutoDiff(reshapedN, ImmutableSet.of(m, alpha));

        Assert.assertArrayEquals(new int[]{4, 1, 2, 2}, forward.withRespectTo(m).getShape());
        Assert.assertArrayEquals(new int[]{4, 1, 2, 2}, backward.withRespectTo(m).getShape());

        double[] expectedPartial = N.getDualNumber().getPartialDerivatives().withRespectTo(m).asFlatDoubleArray();

        Assert.assertArrayEquals(expectedPartial, forward.withRespectTo(m).asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(expectedPartial, backward.withRespectTo(m).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void flatPartialDerivativeIsTheSameAfterReshape() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(a);
        DualNumber NDual = N.getDualNumber();

        DoubleTensor dNdm = NDual.getPartialDerivatives().withRespectTo(m);
        DoubleTensor dNda = NDual.getPartialDerivatives().withRespectTo(a);

        double[] nWrtMpartialsBeforeReshape = dNdm.asFlatDoubleArray();
        double[] nWrtApartialsBeforeReshape = dNda.asFlatDoubleArray();

        ReshapeVertex reshapedN = new ReshapeVertex(N, 4, 1);
        DoubleTensor reshapedPartialWrtM = reshapedN.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor reshapedPartialWrtA = reshapedN.getDualNumber().getPartialDerivatives().withRespectTo(a);

        Assert.assertArrayEquals(nWrtMpartialsBeforeReshape, reshapedPartialWrtM.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(nWrtApartialsBeforeReshape, reshapedPartialWrtA.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void partialCorrectlyFlowsThroughReshape() {
        DoubleVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex C = A.plus(B);

        DoubleVertex D = C.reshape(4, 1);

        DoubleVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 4, 1));

        DoubleVertex F = D.times(E);

        PartialDerivatives forward = F.getDualNumber().getPartialDerivatives();
        PartialDerivatives backward = Differentiator.reverseModeAutoDiff(F, ImmutableSet.of(A, B));

        Assert.assertArrayEquals(new int[]{4, 1, 2, 2}, forward.withRespectTo(A).getShape());
        Assert.assertArrayEquals(forward.withRespectTo(A).asFlatDoubleArray(), backward.withRespectTo(A).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void partialCorrectlyFlowsThroughTwoReshapes() {
        DoubleVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex C = A.plus(B);

        DoubleVertex D = C.reshape(4, 1);
        DoubleVertex E = D.reshape(1, 4);

        PartialDerivatives forward = E.getDualNumber().getPartialDerivatives();
        PartialDerivatives backward = Differentiator.reverseModeAutoDiff(E, ImmutableSet.of(A, B));

        Assert.assertArrayEquals(new int[]{1, 4, 2, 2}, forward.withRespectTo(A).getShape());
        Assert.assertArrayEquals(forward.withRespectTo(A).asFlatDoubleArray(), backward.withRespectTo(A).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new int[]{4, 4}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(1.5).reshape(2, 2, 2, 2);

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 10.0, 1e-10, true);
    }

}
