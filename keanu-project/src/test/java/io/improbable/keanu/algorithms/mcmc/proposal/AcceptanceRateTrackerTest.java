package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.mock;

public class AcceptanceRateTrackerTest {

    private Vertex vertex1 = mock(Vertex.class);
    private Vertex vertex2 = mock(Vertex.class);
    Proposal proposal;
    private AcceptanceRateTracker acceptanceRateTracker;

    @Before
    public void createTracker() throws Exception {
        acceptanceRateTracker = new AcceptanceRateTracker();
    }

    @Test
    public void theAcceptanceRateForAnUnrecognisedSetOfVerticesIsNaN() {
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(Double.NaN));
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
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(Double.NaN));

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
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(Double.NaN));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(Double.NaN));

        proposal = new Proposal();
        proposal.addListener(acceptanceRateTracker);
        proposal.setProposal(vertex1, 3.);
        proposal.setProposal(vertex2, 4.);
        proposal.apply();
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1, vertex2)), equalTo(0.5));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex1)), equalTo(Double.NaN));
        assertThat(acceptanceRateTracker.getAcceptanceRate(ImmutableSet.of(vertex2)), equalTo(Double.NaN));
    }
}