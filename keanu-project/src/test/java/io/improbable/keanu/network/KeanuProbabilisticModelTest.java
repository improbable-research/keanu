package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class KeanuProbabilisticModelTest {

    private GaussianVertex A;
    private GaussianVertex B;
    private DoubleVertex C;
    private DoubleVertex D;

    private DoubleTensor initialA;
    private DoubleTensor initialB;
    private DoubleTensor observationD;

    @Parameterized.Parameters(name = "{index}: Test with A={0}, B={1}, observed sum D:{2}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {DoubleTensor.scalar(2.0), DoubleTensor.scalar(3.0), DoubleTensor.scalar(6.0)},
            {DoubleTensor.create(2.0, 3.0), DoubleTensor.create(3.0, 1.0), DoubleTensor.create(6.0, 4.0)},
        });
    }

    public KeanuProbabilisticModelTest(DoubleTensor initialA, DoubleTensor initialB, DoubleTensor observationD) {

        this.initialA = initialA;
        this.initialB = initialB;
        this.observationD = observationD;

        A = new GaussianVertex(initialA.getShape(), 0.0, 1.0);
        B = new GaussianVertex(initialB.getShape(), 0.0, 1.0);
        C = A.plus(B);
        D = new GaussianVertex(C, 1.0);

        A.setValue(initialA);
        B.setValue(initialB);
        D.observe(observationD);
    }

    @Test
    public void canCalculateLogProbOnKeanuProbabilisticModel() {
        ProbabilisticModel probabilisticModel = new KeanuProbabilisticModel(D.getConnectedGraph());
        canCalculateLogProb(probabilisticModel);
    }

    @Test
    public void canCalculateLogProbOnKeanuProbabilisticModelWithGradient() {
        ProbabilisticModel probabilisticModel = new KeanuProbabilisticModelWithGradient(D.getConnectedGraph());
        canCalculateLogProb(probabilisticModel);
    }

    @Test
    public void canCalculateLogLikelihoodOnKeanuProbabilisticModel() {
        ProbabilisticModel probabilisticModel = new KeanuProbabilisticModel(D.getConnectedGraph());
        canCalculateLogLikelihood(probabilisticModel);
    }

    @Test
    public void canCalculateLogLikelihoodOnKeanuProbabilisticModelWithGradient() {
        ProbabilisticModel probabilisticModel = new KeanuProbabilisticModelWithGradient(D.getConnectedGraph());
        canCalculateLogLikelihood(probabilisticModel);
    }

    public void canCalculateLogProb(ProbabilisticModel probabilisticModel) {

        double defaultLogProb = probabilisticModel.logProb();

        double logProb = probabilisticModel.logProb(ImmutableMap.of(
            A.getId(), initialA,
            B.getId(), initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogProb();
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        DoubleTensor newA = KeanuRandom.getDefaultRandom().nextDouble(initialA.getShape());
        double postUpdateLogProb = probabilisticModel.logProb(ImmutableMap.of(
            A.getId(), newA
        ));

        double expectedPostUpdateLogProb = expectedLogProb();

        assertEquals(expectedPostUpdateLogProb, postUpdateLogProb, 1e-5);
    }

    public void canCalculateLogLikelihood(ProbabilisticModel probabilisticModel) {

        double defaultLogProb = probabilisticModel.logLikelihood();

        double logProb = probabilisticModel.logLikelihood(ImmutableMap.of(
            A.getId(), initialA,
            B.getId(), initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogLikelihood();
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        DoubleTensor newA = KeanuRandom.getDefaultRandom().nextDouble(initialA.getShape());
        double postUpdateLogProb = probabilisticModel.logLikelihood(ImmutableMap.of(
            A.getId(), newA
        ));

        double expectedPostUpdateLogProb = expectedLogLikelihood();

        assertEquals(expectedPostUpdateLogProb, postUpdateLogProb, 1e-5);
    }

    private double expectedLogLikelihood() {
        return ProbabilityCalculator.calculateLogProbFor(ImmutableList.of(D));
    }

    private double expectedLogProb() {
        return ProbabilityCalculator.calculateLogProbFor(ImmutableList.of(D, A, B));
    }

}
