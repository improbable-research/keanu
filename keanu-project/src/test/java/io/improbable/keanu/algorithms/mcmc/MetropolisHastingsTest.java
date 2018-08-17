package io.improbable.keanu.algorithms.mcmc;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class MetropolisHastingsTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesContinuousPrior() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesNet,
            Arrays.asList(A, B),
            100000
        );

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar();

        double actual = averagePosteriorA + averagePosteriorB;
        assertEquals(44.0, actual, 0.1);
    }

    @Test
    public void samplesContinuousTensorPrior() {

        int[] shape = new int[]{1, 1};
        DoubleVertex A = new GaussianVertex(shape, 20.0, 1.0);
        DoubleVertex B = new GaussianVertex(shape, 20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);
        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesNet,
            Arrays.asList(A, B),
            100000
        );

        DoubleTensor averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages();
        DoubleTensor averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages();

        DoubleTensor allActuals = averagePosteriorA.plus(averagePosteriorB);

        for (double actual : allActuals.asFlatDoubleArray()) {
            assertEquals(44.0, actual, 0.1);
        }
    }

    @Test
    public void samplesSimpleDiscretePrior() {

        BernoulliVertex A = new BernoulliVertex(0.5);

        DoubleVertex B = If.isTrue(A)
            .then(0.9)
            .orElse(0.1);

        BernoulliVertex C = new BernoulliVertex(B);

        C.observe(true);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, C));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesNet,
            Collections.singletonList(A),
            10000
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.9, postProbTrue, 0.01);
    }

    @Test
    public void samplesComplexDiscretePrior() {

        BernoulliVertex A = new BernoulliVertex(0.5);
        BernoulliVertex B = new BernoulliVertex(0.5);

        BoolVertex C = A.or(B);

        DoubleVertex D = If.isTrue(C)
            .then(0.9)
            .orElse(0.1);

        BernoulliVertex E = new BernoulliVertex(D);

        E.observe(true);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, C, D, E));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesNet,
            Collections.singletonList(A),
            100000
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.643, postProbTrue, 0.01);
    }

    @Test
    public void samplesFromPriorWithObservedDeterministic() {

        BernoulliVertex A = new BernoulliVertex(0.5);
        BernoulliVertex B = new BernoulliVertex(0.5);
        BoolVertex C = A.or(B);
        C.observe(false);

        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            net,
            Collections.singletonList(A),
            10000
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.0, postProbTrue, 0.01);
    }

    @Test
    public void doesNotDoExtraWorkOnRejectionWhenRejectionCacheEnabled() {
        AtomicInteger n = new AtomicInteger(0);

        DoubleVertex start = new GaussianVertex(new int[]{1, 3}, 0, 1);

        DoubleVertex blackBox = start.lambda(
            (startValue) -> {
                n.incrementAndGet();
                return startValue.plus(1);
            },
            null
        );

        DoubleVertex pluck0 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(0)), null);
        DoubleVertex pluck1 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(1)), null);
        DoubleVertex pluck2 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(2)), null);

        GaussianVertex out1 = new GaussianVertex(pluck0, 1);
        GaussianVertex out2 = new GaussianVertex(pluck1, 1);
        GaussianVertex out3 = new GaussianVertex(pluck2, 1);

        out1.observe(0);
        out2.observe(0);
        out3.observe(0);

        int sampleCount = 100;
        BayesianNetwork network = new BayesianNetwork(start.getConnectedGraph());

        MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            network,
            network.getLatentVertices(),
            sampleCount
        );

        assertEquals(sampleCount + 1, n.get());
    }

}
