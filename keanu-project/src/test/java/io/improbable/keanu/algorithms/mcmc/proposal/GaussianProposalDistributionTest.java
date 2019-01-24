package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.List;
import java.util.Set;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class GaussianProposalDistributionTest {

    public Proposal proposal;
    DoubleTensor currentState = DoubleTensor.create(4.2, 42.0).transpose();
    DoubleTensor proposedState = DoubleTensor.create(4.3, 43.0).transpose();

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Mock
    public GaussianVertex vertex1;
    @Mock
    public GaussianVertex vertex2;
    private GaussianProposalDistribution proposalDistribution;
    private DoubleTensor sigma;

    @Before
    public void setUpProposalDistribution() throws Exception {
        sigma = DoubleTensor.scalar(1.);
        proposalDistribution = new GaussianProposalDistribution(sigma);
    }

    @Before
    public void setRandomSeed() throws Exception {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Before
    public void setUpProposal() throws Exception {
        when(vertex1.getValue()).thenReturn(DoubleTensor.scalar(currentState.getValue(0)));
        when(vertex2.getValue()).thenReturn(DoubleTensor.scalar(currentState.getValue(1)));

        when(vertex1.getShape()).thenReturn(new long[] {});
        when(vertex2.getShape()).thenReturn(new long[] {});

        proposal = new Proposal();
        proposal.setProposal(vertex1, DoubleTensor.scalar(proposedState.getValue(0)));
        proposal.setProposal(vertex2, DoubleTensor.scalar(proposedState.getValue(1)));
    }

    @Test
    public void theLogProbAtToIsGaussianAroundTheGivenPoint() {
        double logProb = proposalDistribution.logProbAtToGivenFrom(proposal);
        DoubleTensor expectedLogProb = Gaussian.withParameters(currentState, sigma).logProb(proposedState);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void theLogProbAtFromIsGaussianAroundTheGivenPoint() {
        double logProb = proposalDistribution.logProbAtFromGivenTo(proposal);
        DoubleTensor expectedLogProb = Gaussian.withParameters(proposedState, sigma).logProb(currentState);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void youCanAddProposalListeners() {
        ProposalListener listener1 = mock(ProposalListener.class);
        ProposalListener listener2 = mock(ProposalListener.class);
        List<ProposalListener> listeners = ImmutableList.of(listener1, listener2);
        proposalDistribution = new GaussianProposalDistribution(sigma, listeners);
        Set<Variable> variables = ImmutableSet.of(vertex1, vertex2);
        Proposal proposal = proposalDistribution.getProposal(variables, KeanuRandom.getDefaultRandom());
        verify(listener1).onProposalCreated(proposal);
        verify(listener2).onProposalCreated(proposal);
        proposalDistribution.onProposalRejected();
        verify(listener1).onProposalRejected(proposal);
        verify(listener2).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener1, listener2);
    }

    @Test
    public void itThrowsIfYouUseItOnADiscreteVariable() {
        thrown.expect(IllegalStateException.class);
        thrown.expectMessage("Gaussian proposal function cannot be used for discrete variable");
        PoissonVertex poisson = new PoissonVertex(1.);
        proposalDistribution.getProposal(ImmutableSet.of(poisson), KeanuRandom.getDefaultRandom());
    }
}
