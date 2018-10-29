package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Rule;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class MetropolisHastingsTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void samplesContinuousPrior() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100);

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

        long[] shape = new long[]{1, 1};
        DoubleVertex A = new GaussianVertex(shape, 20.0, 1.0);
        DoubleVertex B = new GaussianVertex(shape, 20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);
        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100);

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
        bayesNet.probeForNonZeroProbability(100);

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
        bayesNet.probeForNonZeroProbability(100);

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
        net.probeForNonZeroProbability(100);

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            net,
            Collections.singletonList(A),
            10000
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.0, postProbTrue, 0.01);
    }

    @Test
    public void youCanUseAGaussianProposal() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100);

        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(1.));
        MetropolisHastings metropolisHastings = MetropolisHastings.builder()
            .proposalDistribution(proposalDistribution)
            .build();

        NetworkSamples posteriorSamples =  metropolisHastings.getPosteriorSamples(
            bayesNet,
            Arrays.asList(A, B),
            1000
        );

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar();

        double actual = averagePosteriorA + averagePosteriorB;
        assertEquals(44.0, actual, 0.1);
    }

    @Test
    public void doesNotDoExtraWorkOnRejectionWhenRejectionCacheEnabled() {
        AtomicInteger n = new AtomicInteger(0);

        DoubleVertex start = new GaussianVertex(new long[]{1, 3}, 0, 1);

        DoubleVertex blackBox = start.lambda(
            (startValue) -> {
                n.incrementAndGet();
                return startValue.plus(1);
            },
            null,
            null
        );

        DoubleVertex pluck0 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(0)), null, null);
        DoubleVertex pluck1 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(1)), null, null);
        DoubleVertex pluck2 = blackBox.lambda(bb -> DoubleTensor.scalar(bb.getValue(2)), null, null);

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

    @Test
    public void canDefaultToSettingsInBuilderAndIsConfigurableAfterBuilding() {

        GaussianVertex A = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100);

        MetropolisHastings algo = MetropolisHastings.builder()
            .useCacheOnRejection(false)
            .build();

        assertNotNull(algo.getProposalDistribution());
        assertNotNull(algo.getRandom());
        assertNotNull(algo.getVariableSelector());

        NetworkSamples posteriorSamples = algo.getPosteriorSamples(
            net,
            net.getLatentVertices(),
            2
        );

        algo.setVariableSelector(null);
        assertNull(algo.getVariableSelector());

        assertFalse(posteriorSamples.get(A).asList().isEmpty());
    }

    @Test
    public void doesNotStoreSamplesThatWillBeDropped() {

        MetropolisHastings algo = MetropolisHastings.withDefaultConfig();

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 2;
        GaussianVertex A = new GaussianVertex(0, 1);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        NetworkSamples samples = MetropolisHastings.withDefaultConfig().generatePosteriorSamples(network, network.getLatentVertices())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .generate(sampleCount);

        assertEquals((sampleCount - dropCount) / downSampleInterval, samples.size());
        assertEquals(0.0, samples.getDoubleTensorSamples(A).getAverages().scalar(), 0.1);
    }

    @Test
    public void canStreamSamples() {

        MetropolisHastings algo = MetropolisHastings.withDefaultConfig();

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 1;
        GaussianVertex A = new GaussianVertex(0, 1);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        double averageA = algo.generatePosteriorSamples(network, network.getLatentVertices())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .stream()
            .limit(sampleCount)
            .mapToDouble(networkState -> networkState.get(A).scalar())
            .average().getAsDouble();

        assertEquals(0.0, averageA, 0.1);
    }

}
