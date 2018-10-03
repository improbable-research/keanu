package io.improbable.keanu.algorithms.mcmc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import java.util.Collections;
import java.util.Set;
import lombok.AllArgsConstructor;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

public class MetropolisHastingsStepTest {

    @Rule public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom alwaysAccept;
    private KeanuRandom alwaysReject;

    @Before
    public void setup() {
        alwaysAccept = mock(KeanuRandom.class);
        when(alwaysAccept.nextDouble()).thenReturn(0.0);
        when(alwaysAccept.nextGaussian(any())).thenReturn(DoubleTensor.ZERO_SCALAR);

        alwaysReject = mock(KeanuRandom.class);
        when(alwaysReject.nextDouble()).thenReturn(1.0);
        when(alwaysReject.nextGaussian(any())).thenReturn(DoubleTensor.ZERO_SCALAR);
    }

    @Test
    public void doesCalculateCorrectLogProbAfterAcceptingStep() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(1.0);
        DoubleVertex B = A.times(2);
        DoubleVertex observedB = new GaussianVertex(B, 1);
        observedB.observe(5);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        double logProbBeforeStep = network.getLogOfMasterP();

        MetropolisHastingsStep mhStep =
                new MetropolisHastingsStep(
                        network.getLatentVertices(),
                        ProposalDistribution.usePrior(),
                        true,
                        alwaysAccept);

        MetropolisHastingsStep.StepResult result =
                mhStep.step(Collections.singleton(A), logProbBeforeStep);

        assertTrue(result.isAccepted());
        assertEquals(network.getLogOfMasterP(), result.getLogProbabilityAfterStep(), 1e-10);
    }

    @Test
    public void doesAllowCustomProposalDistribution() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(0.0);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        MetropolisHastingsStep mhStep =
                stepFunctionWithConstantProposal(network, 1.0, alwaysAccept);

        MetropolisHastingsStep.StepResult result =
                mhStep.step(Collections.singleton(A), network.getLogOfMasterP());

        assertTrue(result.isAccepted());
        assertEquals(1.0, A.getValue(0), 1e-10);
    }

    @Test
    public void doesRejectOnImpossibleProposal() {
        DoubleVertex A = new UniformVertex(0, 1);
        A.setValue(0.5);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(network, -1, alwaysAccept);

        MetropolisHastingsStep.StepResult result =
                mhStep.step(Collections.singleton(A), network.getLogOfMasterP());

        assertFalse(result.isAccepted());
        assertEquals(0.5, A.getValue(0), 1e-10);
    }

    @Test
    public void doesRejectWhenRejectProbabilityIsOne() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(0.5);
        DoubleVertex B = A.times(2);
        DoubleVertex C = new GaussianVertex(B, 1);
        C.observe(5.0);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(network, 10, alwaysReject);

        MetropolisHastingsStep.StepResult result =
                mhStep.step(Collections.singleton(A), network.getLogOfMasterP());

        assertFalse(result.isAccepted());
        assertEquals(0.5, A.getValue(0), 1e-10);
    }

    private MetropolisHastingsStep stepFunctionWithConstantProposal(
            BayesianNetwork network, double constant, KeanuRandom random) {
        return new MetropolisHastingsStep(
                network.getLatentVertices(), constantProposal(constant), true, random);
    }

    private ProposalDistribution constantProposal(double constant) {
        return new ConstantProposalDistribution(constant);
    }

    @AllArgsConstructor
    private static class ConstantProposalDistribution extends PriorProposalDistribution {

        private final double constant;

        @Override
        public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
            Proposal proposal = new Proposal();
            vertices.forEach(vertex -> proposal.setProposal(vertex, DoubleTensor.scalar(constant)));
            return proposal;
        }
    }
}
