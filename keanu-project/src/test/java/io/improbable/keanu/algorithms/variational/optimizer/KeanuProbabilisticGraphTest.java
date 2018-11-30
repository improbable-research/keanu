package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class KeanuProbabilisticGraphTest {

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

    public KeanuProbabilisticGraphTest(DoubleTensor initialA, DoubleTensor initialB, DoubleTensor observationD) {

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
    public void canCalculateLogProbOnKeanuProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = new KeanuProbabilisticGraph(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbOnKeanuProbabilisticWithGradientGraph() {
        ProbabilisticGraph probabilisticGraph = new KeanuProbabilisticWithGradientGraph(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    private double expectedLogProb(DoubleTensor a, DoubleTensor b, DoubleTensor d) {
        NormalDistribution latents = new NormalDistribution(0.0, 1.0);

        double aLogProb = Arrays.stream(a.asFlatDoubleArray())
            .map(latents::logDensity)
            .sum();

        double bLogProb = Arrays.stream(b.asFlatDoubleArray())
            .map(latents::logDensity)
            .sum();

        double[] abSum = a.plus(b).asFlatDoubleArray();
        double[] dFlat = d.asFlatDoubleArray();

        double dLogProb = 0;
        for (int i = 0; i < dFlat.length; i++) {
            dLogProb += new NormalDistribution(abSum[i], 1.0).logDensity(dFlat[i]);
        }

        return aLogProb + bLogProb + dLogProb;
    }

    public void canCalculateLogProb(ProbabilisticGraph probabilisticGraph) {

        double defaultLogProb = probabilisticGraph.logProb();

        double logProb = probabilisticGraph.logProb(ImmutableMap.of(
            A, initialA,
            B, initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogProb(initialA, initialB, observationD);
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        DoubleTensor newA = KeanuRandom.getDefaultRandom().nextDouble(initialA.getShape());
        double postUpdateLogProb = probabilisticGraph.logProb(ImmutableMap.of(
            A, newA
        ));

        double expectedPostUpdateLogProb = expectedLogProb(newA, initialB, observationD);

        assertEquals(expectedPostUpdateLogProb, postUpdateLogProb, 1e-5);
    }

}
