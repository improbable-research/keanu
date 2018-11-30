package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.mock;


public class AcceptanceRateTrackerTest {

    private Vertex vertex1 = mock(Vertex.class);
    private Vertex vertex2 = mock(Vertex.class);
    Proposal proposal;
    private AcceptanceRateTracker acceptanceRateTracker;

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Before
    public void createTracker() throws Exception {
        acceptanceRateTracker = new AcceptanceRateTracker();
    }

    @Test
    public void itThrowsIfYouAskForTheAcceptanceRateForAnUnrecognisedSetOfVertices() {
        expectedException.expect(IllegalStateException.class);
        expectedException.expectMessage("No proposals have been registered for [Mock for Vertex, hashCode:");
        acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1));
    }

    @Test
    public void youCanTrackTheAcceptanceRateForASingleVertex() {
        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 1.);
        proposal.apply();
        proposal.reject();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.));

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 2.);
        proposal.apply();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.5));
    }


    @Test
    public void youCanTrackTheAcceptanceRateForDifferentSetsOfVertices() {
        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 1.);
        proposal.apply();
        proposal.reject();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.));
        expectRateToBeMissing(vertex2);

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex2, 1.);
        proposal.apply();
        proposal.reject();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(0.));

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 2.);
        proposal.apply();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(0.));

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex2, 2.);
        proposal.apply();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(0.5));
    }

    @Test
    public void youCanTrackTheAcceptanceRateOfASetOfMultipleVertices() {
        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 1.);
        proposal.setProposal(vertex2, 2.);
        proposal.apply();
        proposal.reject();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1, vertex2)), equalTo(0.));
        expectRateToBeMissing(vertex1);
        expectRateToBeMissing(vertex2);

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 3.);
        proposal.setProposal(vertex2, 4.);
        proposal.apply();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1, vertex2)), equalTo(0.5));
        expectRateToBeMissing(vertex1);
        expectRateToBeMissing(vertex2);
    }

    private void expectRateToBeMissing(Vertex vertex) {
        try {
            double acceptanceRate = acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex));
            throw new RuntimeException(String.format("Expected rate for %s to be missing but got %.2f", vertex, acceptanceRate));
        } catch (IllegalStateException e) {
            // pass
        }
    }
}