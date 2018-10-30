package io.improbable.keanu.backend;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
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

    private static final String A_LABEL = "A";
    private static final String B_LABEL = "B";
    private static final String C_LABEL = "C";
    private static final String D_LABEL = "D";

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0).setLabel(A_LABEL);
        B = new GaussianVertex(0.0, 1.0).setLabel(B_LABEL);
        C = A.plus(B).setLabel(C_LABEL);
        D = new GaussianVertex(C, 1.0).setLabel(D_LABEL);
        D.observe(6.0);
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

    private double expectedLogProb(double a, double b) {
        NormalDistribution latents = new NormalDistribution(0.0, 1.0);
        NormalDistribution observation = new NormalDistribution(a + b, 1.0);
        return latents.logDensity(a) + latents.logDensity(b) + observation.logDensity(6.0);
    }

    public void canCalculateLogProb(ProbabilisticGraph probabilisticGraph) {

        double a = 2;
        double b = 3;

        double logProb = probabilisticGraph.logProb(ImmutableMap.of(
            A_LABEL, DoubleTensor.scalar(a),
            B_LABEL, DoubleTensor.scalar(b)
        ));

        assertEquals(expectedLogProb(a, b), logProb, 1e-5);

        double logProb2 = probabilisticGraph.logProb(ImmutableMap.of(
            A_LABEL, DoubleTensor.scalar(3)
        ));

        assertEquals(expectedLogProb(3, b), logProb2, 1e-5);
    }

    public void canConvertSimpleNetworkAndTakeSample(ProbabilisticGraph probabilisticGraph) {

        double a = 2;
        double b = 3;

        LogProbWithSample logProbWithSample = probabilisticGraph.logProbWithSample(ImmutableMap.of(
            A_LABEL, DoubleTensor.scalar(a),
            B_LABEL, DoubleTensor.scalar(b)
        ), ImmutableList.of(A_LABEL, B_LABEL, C_LABEL));

        assertEquals(expectedLogProb(a, b), logProbWithSample.getLogProb(), 1e-5);

        Map<String, ?> sample = logProbWithSample.getSample();
        assertEquals(a, ((DoubleTensor) sample.get(A_LABEL)).scalar());
        assertEquals(b, ((DoubleTensor) sample.get(B_LABEL)).scalar());
        assertEquals(a + b, ((DoubleTensor) sample.get(C_LABEL)).scalar());
    }
}
