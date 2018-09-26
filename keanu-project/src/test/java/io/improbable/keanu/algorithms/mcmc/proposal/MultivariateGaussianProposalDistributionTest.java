package io.improbable.keanu.algorithms.mcmc.proposal;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Covariance;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

@RunWith(MockitoJUnitRunner.class)
public class MultivariateGaussianProposalDistributionTest {

    private Covariance covariance;
    public Proposal proposal;
    VertexId vertexId1 = new VertexId();
    VertexId vertexId2 = new VertexId();
    DoubleTensor currentState = DoubleTensor.create(4.2, 42.0).transpose();
    DoubleTensor proposedState = DoubleTensor.create(4.3, 43.0).transpose();
    private MultivariateGaussian distribution;
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Mock
    public Vertex vertex1;
    @Mock
    public Vertex vertex2;
    private MultivariateGaussianProposalDistribution proposalDistribution;

    @Before
    public void setUpProposalDistribution() throws Exception {
        DoubleTensor covarianceMatrix = DoubleTensor.create(new double[]{
                1., 0.1,
                0.1, 3.},
            2, 2);

        covariance = new Covariance(covarianceMatrix, vertexId1, vertexId2);
        proposalDistribution = new MultivariateGaussianProposalDistribution(covariance);
    }

    @Before
    public void setRandomSeed() throws Exception {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Before
    public void setUpMocks() throws Exception {
        when(vertex1.getId()).thenReturn(vertexId1);
        when(vertex2.getId()).thenReturn(vertexId2);
        when(vertex1.getValue()).thenReturn(DoubleTensor.scalar(currentState.getValue(0)));
        when(vertex2.getValue()).thenReturn(DoubleTensor.scalar(currentState.getValue(1)));

        proposal = new Proposal();
        proposal.setProposal(vertex1, DoubleTensor.scalar(proposedState.getValue(0)));
        proposal.setProposal(vertex2, DoubleTensor.scalar(proposedState.getValue(1)));
    }

    @Test
    public void theLogProbAtToIsMultinomialAroundTheGivenPoint() {
        double logProb = proposalDistribution.logProbAtToGivenFrom(proposal);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(currentState, covariance.asTensor()).logProb(proposedState);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void theLogProbAtFromIsMultinomialAroundTheGivenPoint() {
        double logProb = proposalDistribution.logProbAtFromGivenTo(proposal);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(proposedState, covariance.asTensor()).logProb(currentState);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }
}