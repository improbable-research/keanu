package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MetropolisHastingsStepTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom alwaysAccept;
    private KeanuRandom alwaysReject;

    @Before
    public void setup() {
        alwaysAccept = mock(KeanuRandom.class);
        when(alwaysAccept.nextDouble()).thenReturn(0.0);

        alwaysReject = mock(KeanuRandom.class);
        when(alwaysReject.nextDouble()).thenReturn(1.0);
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

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            network.getLatentVertices(),
            ProposalDistribution.usePrior,
            true
        );

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            logProbBeforeStep,
            KeanuRandom.getDefaultRandom()
        );

        assertEquals(network.getLogOfMasterP(), result.getLogProbAfterStep(), 1e-10);
        assertTrue(result.isAccepted());
    }

    @Test
    public void doesAllowCustomProposalDistribution() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(0.0);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(network, 1.0);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            network.getLogOfMasterP(),
            alwaysAccept
        );

        assertTrue(result.isAccepted());
        assertEquals(1.0, A.getValue(0), 1e-10);
    }

    @Test
    public void doesRejectOnImpossibleProposal() {
        DoubleVertex A = new UniformVertex(0, 1);
        A.setValue(0.5);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(network, -1);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            network.getLogOfMasterP(),
            alwaysAccept
        );

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

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(network, 10);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            network.getLogOfMasterP(),
            alwaysReject
        );

        assertFalse(result.isAccepted());
        assertEquals(0.5, A.getValue(0), 1e-10);
    }

    private MetropolisHastingsStep stepFunctionWithConstantProposal(BayesianNetwork network, double constant) {
        return new MetropolisHastingsStep(
            network.getLatentVertices(),
            constantProposal(constant),
            true
        );
    }

    private ProposalDistribution constantProposal(double constant) {
        return (vertices, random) -> {
            Proposal proposal = new Proposal();
            vertices.forEach(vertex -> proposal.setProposal(vertex, DoubleTensor.scalar(constant)));
            return proposal;
        };
    }

}
