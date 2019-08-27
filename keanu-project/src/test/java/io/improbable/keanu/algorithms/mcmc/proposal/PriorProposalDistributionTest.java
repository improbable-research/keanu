package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class PriorProposalDistributionTest {

    public GaussianVertex vertex1;
    public GaussianVertex vertex2;
    private PriorProposalDistribution proposalDistribution;

    @Before
    public void setUpProposalDistribution() {
        vertex1 = new GaussianVertex(0, 1);
        vertex2 = new GaussianVertex(0, 1);
        proposalDistribution = new PriorProposalDistribution();
    }

    @Before
    public void setRandomSeed() {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Test
    public void youCanAddProposalListeners() {
        ProposalListener listener1 = mock(ProposalListener.class);
        ProposalListener listener2 = mock(ProposalListener.class);
        List<ProposalListener> listeners = ImmutableList.of(listener1, listener2);
        proposalDistribution = new PriorProposalDistribution(listeners);
        Set<Variable> variables = ImmutableSet.of(vertex1, vertex2);
        Proposal proposal = proposalDistribution.getProposal(variables, KeanuRandom.getDefaultRandom());
        verify(listener1).onProposalCreated(proposal);
        verify(listener2).onProposalCreated(proposal);
        proposalDistribution.onProposalRejected();
        verify(listener1).onProposalRejected(proposal);
        verify(listener2).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener1, listener2);
    }
}
