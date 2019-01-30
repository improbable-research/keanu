package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.testcases.MCMCTestCase;
import io.improbable.keanu.algorithms.mcmc.testcases.MultiVariateDiscreteTestCase;
import io.improbable.keanu.algorithms.mcmc.testcases.SingleVariateDiscreteTestCase;
import io.improbable.keanu.algorithms.mcmc.testcases.SumGaussianTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

public class MetropolisHastingsTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void samplesContinuousPriorSingleVariableSelected() {

        MCMCTestCase testCase = new SumGaussianTestCase();

        BayesianNetwork bayesNet = testCase.getModel();
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = MetropolisHastings.builder()
            .proposalDistribution(new PriorProposalDistribution(model.getLatentVertices()))
            .rejectionStrategy(new RollBackToCachedValuesOnRejection(model.getLatentVertices()))
            .variableSelector(MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR)
            .build()
            .getPosteriorSamples(
                model,
                model.getLatentVertices(),
                5000
            );

        testCase.assertExpected(posteriorSamples);
    }

    @Test
    public void samplesContinuousPriorAllVariablesSelected() {

        MCMCTestCase testCase = new SumGaussianTestCase();

        BayesianNetwork bayesNet = testCase.getModel();
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = MetropolisHastings.builder()
            .proposalDistribution(new PriorProposalDistribution(model.getLatentVertices()))
            .rejectionStrategy(new RollBackToCachedValuesOnRejection(model.getLatentVertices()))
            .variableSelector(MHStepVariableSelector.FULL_VARIABLE_SELECTOR)
            .build()
            .getPosteriorSamples(
                model,
                model.getLatentVertices(),
                5000
            );

        testCase.assertExpected(posteriorSamples);
    }

    @Category(Slow.class)
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
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(
            model,
            Arrays.asList(A, B),
            5000
        );

        DoubleTensor averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages();
        DoubleTensor averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages();

        DoubleTensor allActuals = averagePosteriorA.plus(averagePosteriorB);

        for (double actual : allActuals.asFlatDoubleArray()) {
            assertEquals(44.0, actual, 0.1);
        }
    }

    @Test
    public void samplesSimpleDiscretePriorWithDefaults() {

        MCMCTestCase testCase = new SingleVariateDiscreteTestCase();

        BayesianNetwork bayesNet = testCase.getModel();
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(
            model,
            model.getLatentVertices(),
            10000
        );

        testCase.assertExpected(posteriorSamples);
    }

    @Category(Slow.class)
    @Test
    public void samplesComplexDiscretePriorWithDefaults() {

        MCMCTestCase testCase = new MultiVariateDiscreteTestCase();

        BayesianNetwork bayesNet = testCase.getModel();
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(
            model,
            model.getLatentVertices(),
            1000
        );

        testCase.assertExpected(posteriorSamples);
    }

    @Category(Slow.class)
    @Test
    public void samplesComplexDiscreteWithFullVariableSelect() {

        MCMCTestCase testCase = new MultiVariateDiscreteTestCase();

        BayesianNetwork bayesNet = testCase.getModel();

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);

        NetworkSamples posteriorSamples = MetropolisHastings.builder()
            .proposalDistribution(new PriorProposalDistribution(model.getLatentVertices()))
            .rejectionStrategy(new RollBackToCachedValuesOnRejection(model.getLatentVertices()))
            .variableSelector(MHStepVariableSelector.FULL_VARIABLE_SELECTOR)
            .build()
            .getPosteriorSamples(
                model,
                model.getLatentVertices(),
                1000
            );

        testCase.assertExpected(posteriorSamples);
    }

    @Test
    public void samplesFromPriorWithObservedDeterministic() {

        BernoulliVertex A = new BernoulliVertex(0.5);
        BernoulliVertex B = new BernoulliVertex(0.5);
        BooleanVertex C = A.or(B);
        C.observe(false);

        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

        NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(
            model,
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
            .rejectionStrategy(new RollbackAndCascadeOnRejection(bayesNet.getLatentVertices()))
            .build();

        NetworkSamples posteriorSamples = metropolisHastings.getPosteriorSamples(
            new KeanuProbabilisticModel(bayesNet),
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

        MetropolisHastings.builder()
            .proposalDistribution(new PriorProposalDistribution(network.getLatentVertices()))
            .rejectionStrategy(new RollBackToCachedValuesOnRejection(network.getLatentVertices()))
            .build()
            .getPosteriorSamples(
                new KeanuProbabilisticModel(network),
                network.getLatentVertices(),
                sampleCount
            );

        assertEquals(sampleCount + 1, n.get());
    }

    @Test
    public void doesNotStoreSamplesThatWillBeDropped() {

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 2;
        GaussianVertex A = new GaussianVertex(0, 1);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());
        NetworkSamples samples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model)
            .generatePosteriorSamples(model, model.getLatentVariables())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .generate(sampleCount);

        assertEquals((sampleCount - dropCount) / downSampleInterval, samples.size());
        assertEquals(0.0, samples.getDoubleTensorSamples(A).getAverages().scalar(), 0.1);
    }

    @Test
    public void canStreamSamples() {

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 1;
        GaussianVertex A = new GaussianVertex(0, 1);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());
        MetropolisHastings algo = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model);

        double averageA = algo.generatePosteriorSamples(model, model.getLatentVariables())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .stream()
            .limit(sampleCount)
            .mapToDouble(networkState -> networkState.get(A).scalar())
            .average().getAsDouble();

        assertEquals(0.0, averageA, 0.1);
    }

}
