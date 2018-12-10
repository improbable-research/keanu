package io.improbable.keanu.algorithms.mcmc.proposal;


import org.junit.Before;
import org.junit.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class ProposalTest {
    Proposal proposal;

    @Before
    public void createProposal() throws Exception {
        proposal = new Proposal();
    }

    @Test
    public void youCanAddAListener() {
        ProposalListener listener = mock(ProposalListener.class);
        proposal.addListener(listener);
        proposal.apply();
        verify(listener).onProposalApplied(proposal);
        proposal.reject();
        verify(listener).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener);
    }
}