package io.improbable.keanu.algorithms.mcmc.proposal;


import com.google.common.collect.ImmutableList;
import org.junit.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class ProposalTest {

    @Test
    public void youCanAddAListener() {
        ProposalListener listener = mock(ProposalListener.class);
        ProposalNotifier notifier = new ProposalNotifier(ImmutableList.of(listener));
        Proposal proposal = mock(Proposal.class);

        notifier.notifyProposalCreated(proposal);
        verify(listener).onProposalCreated(proposal);
        notifier.notifyProposalRejected();
        verify(listener).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener);
    }
}