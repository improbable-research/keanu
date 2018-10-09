package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Map;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LogProbGradientCalculatorTest {

    @Test
    public void canFindGradientOfSingleVariateGaussianWrtSelf() {
        GaussianVertex A = new GaussianVertex(0, 1);
        A.setValue(0.5);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(A), ImmutableList.of(A));

        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dALogProbWrtAValue = gradient.get(A.getId());

        ContinuousDistribution distribution = Gaussian.withParameters(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);
        DoubleTensor expected = distribution.dLogProb(DoubleTensor.scalar(0.5)).get(Diffs.X).getValue();

        assertThat(dALogProbWrtAValue, equalTo(expected));
    }

    @Test
    public void canFindGradientOfSingleVariantGaussianWrtSingleVariateLatent() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(A, 1);
        B.setValue(0.5);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A));

        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dBLogProbWrtAValue = gradient.get(A.getId());

        DoubleTensor expectedDLogProbWrtA = B.dLogProb(DoubleTensor.scalar(0.5), A).get(A);

        assertThat(dBLogProbWrtAValue, equalTo(expectedDLogProbWrtA));
    }

    @Test
    public void canFindGradientOfSingleVariantGaussianWrtMultivariateLatent() {

        GaussianVertex A = new GaussianVertex(new int[]{3, 2}, 0, 1);
        GaussianVertex B = new GaussianVertex(A, 1);
        DoubleTensor bValue = DoubleTensor.create(new double[]{0.1, 0.2, 0.3, -0.2, -0.5, 0.9}, 3, 2);
        B.setValue(bValue);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A));

        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dBLogProbWrtAValue = gradient.get(A.getId());

        DoubleTensor expectedDLogProbWrtA = B.dLogProb(bValue, A).get(A);

        assertThat(dBLogProbWrtAValue, equalTo(expectedDLogProbWrtA));
    }

    @Test
    public void canFindGradientOfMultivariantGaussianWrtSingleVariateLatent() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(new int[]{3, 2}, A, 1);
        DoubleTensor bValue = DoubleTensor.create(new double[]{0.1, 0.2, 0.3, -0.2, -0.5, 0.9}, 3, 2);
        B.setValue(bValue);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A));

        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dBLogProbWrtAValue = gradient.get(A.getId());

        double expectedDLogProbWrtA = B.dLogProb(bValue, A).get(A).sum();

        assertThat(dBLogProbWrtAValue.scalar(), equalTo(expectedDLogProbWrtA));
        assertThat(dBLogProbWrtAValue.getLength(), equalTo(1L));
    }

    @Test
    public void canFindGradientOfMultivariantGaussianWrtSingleVariateLatentWithOp() {

        GaussianVertex A = new GaussianVertex(0, 1);
        DoubleTensor aValue = DoubleTensor.scalar(0.2);
        A.setValue(aValue);
        GaussianVertex C = new GaussianVertex(new int[]{3, 2}, 0, 1);
        DoubleTensor cValue = DoubleTensor.create(new double[]{-0.1, -0.2, -0.3, 0.2, 0.5, -0.9}, 3, 2);
        C.setValue(cValue);
        DoubleVertex D = A.times(C);
        GaussianVertex B = new GaussianVertex(D, 1);
        DoubleTensor bValue = DoubleTensor.create(new double[]{0.1, 0.2, 0.3, -0.2, -0.5, 0.9}, 3, 2);
        B.setValue(bValue);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A, C));

        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dBLogProbWrtAValue = gradient.get(A.getId());
        DoubleTensor dBLogProbWrtCValue = gradient.get(C.getId());

        double expectedDLogProbWrtA = B.dLogProb(bValue, D).get(D).times(cValue).sum();
        DoubleTensor expectedDLogProbWrtC = B.dLogProb(bValue, D).get(D).times(aValue);

        assertArrayEquals(new int[]{1, 1}, dBLogProbWrtAValue.getShape());
        assertThat(dBLogProbWrtAValue.scalar(), equalTo(expectedDLogProbWrtA));
        assertThat(dBLogProbWrtCValue, equalTo(expectedDLogProbWrtC));
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnLogProbDiffOnNonProbabilistic() {
        DoubleVertex A = ConstantVertex.of(5.0);
        DoubleVertex B = A.times(4);
        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A));
        calculator.getJointLogProbGradientWrtLatents();
    }

    @Test
    public void doesMatchForwardAutodiffWithManyOps() {
        int[] shape = new int[]{2, 2};
        DoubleVertex A = new GaussianVertex(shape, 0, 1);
        A.setValue(DoubleTensor.linspace(0.1, 2, 4).reshape(shape));
        DoubleVertex B = new GaussianVertex(shape, 0, 1);
        B.setValue(DoubleTensor.linspace(0.2, 1, 4).reshape(shape));
        DoubleVertex D = A.atan2(B).sigmoid().times(B);
        DoubleVertex C = A.sin().cos().div(D);
        DoubleVertex E = C.times(D).pow(A).acos();
        DoubleVertex G = E.log().tan().asin().atan();
        DoubleVertex F = D.plus(B).exp();
        DoubleVertex H = G.plus(F).sum();
        GaussianVertex J = new GaussianVertex(H, 1);
        J.observe(0.5);

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(J), ImmutableList.of(A, B));
        Map<VertexId, DoubleTensor> gradient = calculator.getJointLogProbGradientWrtLatents();
        DoubleTensor dJLogProbWrtAValue = gradient.get(A.getId());
        DoubleTensor dJLogProbWrtBValue = gradient.get(B.getId());

        PartialDerivatives dHForward = H.getDerivativeWrtLatents();

        DoubleTensor dHdA = dHForward.withRespectTo(A);
        DoubleTensor dHdB = dHForward.withRespectTo(B);
        DoubleTensor dJLogProbWrtH = J.dLogProbAtValue(H).get(H);

        DoubleTensor expectedDJLogProbWrtAValue = dJLogProbWrtH.times(dHdA).sum(0, 1);
        DoubleTensor expectedDJLogProbWrtBValue = dJLogProbWrtH.times(dHdB).sum(0, 1);

        assertEquals(expectedDJLogProbWrtAValue, dJLogProbWrtAValue);
        assertEquals(expectedDJLogProbWrtBValue, dJLogProbWrtBValue);
    }

}
