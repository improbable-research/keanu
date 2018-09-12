package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

import java.util.Map;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.dual.Diffs;
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

        Gaussian distribution = Gaussian.withParameters(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);
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

        assertThat(dBLogProbWrtAValue, equalTo(DoubleTensor.scalar(expectedDLogProbWrtA)));
        assertThat(dBLogProbWrtCValue, equalTo(expectedDLogProbWrtC));
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnLogProbDiffOnNonProbabilistic() {
        DoubleVertex A = ConstantVertex.of(5.0);
        DoubleVertex B = A.times(4);
        LogProbGradientCalculator calculator = new LogProbGradientCalculator(ImmutableList.of(B), ImmutableList.of(A));
        calculator.getJointLogProbGradientWrtLatents();
    }

}
