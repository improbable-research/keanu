package io.improbable.keanu.backend;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.backend.keanu.KeanuProbabilisticGraph;
import io.improbable.keanu.backend.keanu.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.backend.tensorflow.TensorflowGraphConverter;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class ProbabilisticGraphTest {

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

    public ProbabilisticGraphTest(DoubleTensor initialA, DoubleTensor initialB, DoubleTensor observationD) {

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

    @Test
    public void canCalculateLogLikelihoodOnKeanuProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = new KeanuProbabilisticGraph(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogLikelihood(probabilisticGraph);
    }

    @Test
    public void canCalculateLogLikelihoodOnKeanuProbabilisticWithGradientGraph() {
        ProbabilisticGraph probabilisticGraph = new KeanuProbabilisticWithGradientGraph(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogLikelihood(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbOnTensorflowProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convert(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbOnTensorflowProbabilisticWithGradientGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convertWithGradient(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    @Test
    public void canCalculateLogLikelihoodOnTensorflowProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convert(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogLikelihood(probabilisticGraph);
    }

    @Test
    public void canCalculateLogLikelihoodOnTensorflowProbabilisticWithGradientGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convertWithGradient(new BayesianNetwork(D.getConnectedGraph()));
        canCalculateLogLikelihood(probabilisticGraph);
    }

    public void canCalculateLogProb(ProbabilisticGraph probabilisticGraph) {

        double defaultLogProb = probabilisticGraph.logProb();

        double logProb = probabilisticGraph.logProb(ImmutableMap.of(
            A.getId(), initialA,
            B.getId(), initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogProb();
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        DoubleTensor newA = KeanuRandom.getDefaultRandom().nextDouble(initialA.getShape());
        double postUpdateLogProb = probabilisticGraph.logProb(ImmutableMap.of(
            A.getId(), newA
        ));

        A.setAndCascade(newA);

        double expectedPostUpdateLogProb = expectedLogProb();

        assertEquals(expectedPostUpdateLogProb, postUpdateLogProb, 1e-5);
    }

    public void canCalculateLogLikelihood(ProbabilisticGraph probabilisticGraph) {

        double defaultLogProb = probabilisticGraph.logLikelihood();

        double logProb = probabilisticGraph.logLikelihood(ImmutableMap.of(
            A.getId(), initialA,
            B.getId(), initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogLikelihood();
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        DoubleTensor newA = KeanuRandom.getDefaultRandom().nextDouble(initialA.getShape());
        double postUpdateLogProb = probabilisticGraph.logLikelihood(ImmutableMap.of(
            A.getId(), newA
        ));

        A.setAndCascade(newA);

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
