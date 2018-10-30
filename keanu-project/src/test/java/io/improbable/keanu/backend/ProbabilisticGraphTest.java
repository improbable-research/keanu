package io.improbable.keanu.backend;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.backend.keanu.KeanuGraphConverter;
import io.improbable.keanu.backend.tensorflow.TensorflowGraphConverter;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class ProbabilisticGraphTest {

    private GaussianVertex A;
    private GaussianVertex B;
    private DoubleVertex C;
    private DoubleVertex D;

    private DoubleTensor initialA = DoubleTensor.scalar(2.0);
    private DoubleTensor initialB = DoubleTensor.scalar(3.0);
    private DoubleTensor observationD = DoubleTensor.scalar(6.0);

    private static final String A_LABEL = "A";
    private static final String B_LABEL = "B";
    private static final String C_LABEL = "C";
    private static final String D_LABEL = "D";

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0).setLabel(A_LABEL);
        A.setValue(initialA);
        B = new GaussianVertex(0.0, 1.0).setLabel(B_LABEL);
        B.setValue(initialB);
        C = A.plus(B).setLabel(C_LABEL);
        D = new GaussianVertex(C, 1.0).setLabel(D_LABEL);
        D.observe(observationD);
    }

    @Test
    public void canCalculateLogProbOnKeanuProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = KeanuGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbOnTensorflowProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        canCalculateLogProb(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbAndSampleOnKeanuProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = KeanuGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        canConvertSimpleNetworkAndTakeSample(probabilisticGraph);
    }

    @Test
    public void canCalculateLogProbAndSampleOnTensorflowProbabilisticGraph() {
        ProbabilisticGraph probabilisticGraph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        canConvertSimpleNetworkAndTakeSample(probabilisticGraph);
    }

    private double expectedLogProb(double a, double b, double d) {
        NormalDistribution latents = new NormalDistribution(0.0, 1.0);
        NormalDistribution observation = new NormalDistribution(a + b, 1.0);
        return latents.logDensity(a) + latents.logDensity(b) + observation.logDensity(d);
    }

    public void canCalculateLogProb(ProbabilisticGraph probabilisticGraph) {

        double defaultLogProb = probabilisticGraph.logProb();

        double logProb = probabilisticGraph.logProb(ImmutableMap.of(
            A_LABEL, initialA,
            B_LABEL, initialB
        ));

        assertEquals(defaultLogProb, logProb);

        double expectedInitialLogProb = expectedLogProb(initialA.scalar(), initialB.scalar(), observationD.scalar());
        assertEquals(expectedInitialLogProb, logProb, 1e-5);

        double postUpdateLogProb = probabilisticGraph.logProb(ImmutableMap.of(
            A_LABEL, DoubleTensor.scalar(3)
        ));

        double expectedPostUpdateLogProb = expectedLogProb(3, initialB.scalar(), observationD.scalar());

        assertEquals(expectedPostUpdateLogProb, postUpdateLogProb, 1e-5);
    }

    public void canConvertSimpleNetworkAndTakeSample(ProbabilisticGraph probabilisticGraph) {

        LogProbWithSample logProbWithSample = probabilisticGraph.logProbWithSample(ImmutableMap.of(
            A_LABEL, initialA,
            B_LABEL, initialB
        ), ImmutableList.of(A_LABEL, B_LABEL, C_LABEL));

        double expectedLogProb = expectedLogProb(initialA.scalar(), initialB.scalar(), observationD.scalar());
        assertEquals(expectedLogProb, logProbWithSample.getLogProb(), 1e-5);

        Map<String, ?> sample = logProbWithSample.getSample();
        assertEquals(initialA.scalar(), ((DoubleTensor) sample.get(A_LABEL)).scalar());
        assertEquals(initialB.scalar(), ((DoubleTensor) sample.get(B_LABEL)).scalar());
        assertEquals(initialA.plus(initialB).scalar(), ((DoubleTensor) sample.get(C_LABEL)).scalar());
    }
}
