package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;
import static org.junit.Assert.assertEquals;

public class RejectionSamplerTest {

    private BoolVertex v1;
    private BoolVertex v2;
    private BoolVertex v3;
    private double v1ProbTrueAccordingToBayes;
    private double v2ProbTrueAccordingToBayes;

    @Before
    public void setup() {
        Random random = new Random(1);

        double v1ProbTrue = 0.4;
        double v2ProbTrue = 0.8;

        v1 = new Flip(v1ProbTrue, random);
        v2 = new Flip(v2ProbTrue, random);
        v3 = v1.or(v2);

        double v3ProbTrue = v1ProbTrue + v2ProbTrue - (v1ProbTrue * v2ProbTrue);

        v1ProbTrueAccordingToBayes = v1ProbTrue / v3ProbTrue;
        v2ProbTrueAccordingToBayes = v2ProbTrue / v3ProbTrue;

        v3.observe(true);
    }

    @Test
    public void posteriorProbabilityMatchesBayesRule() {

        double v1ProbTrueAccordingToAlgo = RejectionSampler.getPosteriorProbability(
            asList(v1, v2),
            singletonList(v3),
            v1::getValue,
            10000
        );

        double v2ProbTrueAccordingToAlgo = RejectionSampler.getPosteriorProbability(
            asList(v1, v2),
            singletonList(v3),
            v2::getValue,
            10000
        );

        assertEquals(v1ProbTrueAccordingToBayes, v1ProbTrueAccordingToAlgo, 0.01);
        assertEquals(v2ProbTrueAccordingToBayes, v2ProbTrueAccordingToAlgo, 0.01);
    }

    @Test
    public void posteriorSamplesMatchesBayesRule() {

        NetworkSamples samplesAccordingToAlgo = RejectionSampler.getPosteriorSamples(
            new BayesianNetwork(v1.getConnectedGraph()),
            asList(v1, v2),
            10000
        );

        double v1ProbTrueAccordingToAlgo = samplesAccordingToAlgo.get(v1).probability(sample -> sample);
        double v2ProbTrueAccordingToAlgo = samplesAccordingToAlgo.get(v2).probability(sample -> sample);

        assertEquals(v1ProbTrueAccordingToBayes, v1ProbTrueAccordingToAlgo, 0.01);
        assertEquals(v2ProbTrueAccordingToBayes, v2ProbTrueAccordingToAlgo, 0.01);
    }
}
