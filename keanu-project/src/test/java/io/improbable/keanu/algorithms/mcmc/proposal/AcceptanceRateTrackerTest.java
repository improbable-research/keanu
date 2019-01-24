package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;


public class AcceptanceRateTrackerTest {

    private Vertex vertex1 = mock(Vertex.class);
    private Vertex vertex2 = mock(Vertex.class);
    ProposalNotifier notifier;
    private AcceptanceRateTracker acceptanceRateTracker;

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Before
    public void createTracker() throws Exception {
        acceptanceRateTracker = new AcceptanceRateTracker();
        notifier = new ProposalNotifier(ImmutableList.of(acceptanceRateTracker));
    }

    @Before
    public void setUpMocks() throws Exception {
        when(vertex1.getId()).thenReturn(new VertexId(1));
        when(vertex2.getId()).thenReturn(new VertexId(2));
        when(vertex1.getReference()).thenReturn(new VertexId(1));
        when(vertex2.getReference()).thenReturn(new VertexId(2));
    }

    @Test
    public void itThrowsIfYouAskForTheAcceptanceRateForAnUnrecognisedSetOfVertices() {
        expectedException.expect(IllegalStateException.class);
        expectedException.expectMessage("No proposals have been registered for [1]");
        acceptanceRateTracker.getAcceptanceRate(vertex1.getId());
    }

    @Test
    public void youCanTrackTheAcceptanceRateForASingleVertex() {
        Proposal proposal = new Proposal();
        proposal.setProposal(vertex1, 1.);
        notifier.notifyProposalCreated(proposal);
        notifier.notifyProposalRejected();
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.));

        proposal = new Proposal();
        proposal.setProposal(vertex1, 2.);
        notifier.notifyProposalCreated(proposal);
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.5));
    }


    @Test
    public void youCanTrackTheAcceptanceRateForDifferentSetsOfVertices() {
        Proposal proposal = new Proposal();
        proposal.setProposal(vertex1, 1.);
        notifier.notifyProposalCreated(proposal);
        notifier.notifyProposalRejected();
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.));
        expectRateToBeMissing(vertex2);

        proposal = new Proposal();
        proposal.setProposal(vertex2, 1.);
        notifier.notifyProposalCreated(proposal);
        notifier.notifyProposalRejected();
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.));
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex2.getId()), equalTo(0.));

        proposal = new Proposal();
        proposal.setProposal(vertex1, 2.);
        notifier.notifyProposalCreated(proposal);
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex2.getId()), equalTo(0.));

        proposal = new Proposal();
        proposal.setProposal(vertex2, 2.);
        notifier.notifyProposalCreated(proposal);
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex2.getId()), equalTo(0.5));
    }

    @Test
    public void youCanTrackTheAcceptanceRateOfASetOfMultipleVertices() {
        Proposal proposal = new Proposal();
        proposal.setProposal(vertex1, 1.);
        proposal.setProposal(vertex2, 2.);
        notifier.notifyProposalCreated(proposal);
        notifier.notifyProposalRejected();
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.));
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex2.getId()), equalTo(0.));

        proposal = new Proposal();
        proposal.setProposal(vertex1, 3.);
        proposal.setProposal(vertex2, 4.);
        notifier.notifyProposalCreated(proposal);
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex1.getId()), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(vertex2.getId()), equalTo(0.5));
    }

    private void expectRateToBeMissing(Vertex vertex) {
        try {
            double acceptanceRate = acceptanceRateTracker.getAcceptanceRate(vertex.getId());
            throw new RuntimeException(String.format("Expected rate for %s to be missing but got %.2f", vertex, acceptanceRate));
        } catch (IllegalStateException e) {
            // pass
        }
    }
}