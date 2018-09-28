package io.improbable.keanu.algorithms.mcmc.proposal;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

@RunWith(MockitoJUnitRunner.class)
public class GaussianProposalDistributionTest {

    public Proposal proposal;
    DoubleTensor currentState = DoubleTensor.create(4.2, 42.0).transpose();
    DoubleTensor proposedState = DoubleTensor.create(4.3, 43.0).transpose();

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
}
