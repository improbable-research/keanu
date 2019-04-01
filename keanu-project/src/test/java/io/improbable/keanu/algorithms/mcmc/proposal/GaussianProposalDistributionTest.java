package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class GaussianProposalDistributionTest {

    public Proposal proposal;
    private static final DoubleTensor currentStateForVertex1 = DoubleTensor.create(4.2, 4.7);
    private static final DoubleTensor currentStateForVertex2 = DoubleTensor.create(42.0, 48.0);

    private static final DoubleTensor proposedStateForVertex1 = DoubleTensor.create(4.3, 5.8);
    private static final DoubleTensor proposedStateForVertex2 = DoubleTensor.create(43.0, 32.0);

    private static final DoubleTensor sigmaForVertex1 = DoubleTensor.create(1., 3.);
    private static final DoubleTensor sigmaForVertex2 = DoubleTensor.create(2., 2.);

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Mock
    private GaussianVertex vertex1;
    @Mock
    private GaussianVertex vertex2;

    private GaussianProposalDistribution proposalDistribution;
    private Map<Variable, DoubleTensor> sigmas;

    @Before
    public void setUpProposalDistribution() {
        sigmas = ImmutableMap.of(
            vertex1, sigmaForVertex1,
            vertex2, sigmaForVertex2
        );
        proposalDistribution = new GaussianProposalDistribution(sigmas);
    }

    @Before
    public void setRandomSeed() {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Before
    public void setUpProposal() {
        when(vertex1.getValue()).thenReturn(currentStateForVertex1);
        when(vertex2.getValue()).thenReturn(currentStateForVertex2);

        when(vertex1.getShape()).thenReturn(new long[] {2});
        when(vertex2.getShape()).thenReturn(new long[] {2});

        proposal = new Proposal();
        proposal.setProposal(vertex1, proposedStateForVertex1);
        proposal.setProposal(vertex2, proposedStateForVertex2);
    }

    @Test
    public void theLogProbAtToIsMultivariateGaussian() {
        double logProb = proposalDistribution.logProbAtToGivenFrom(proposal);
        DoubleTensor mu = DoubleTensor.concat(currentStateForVertex1, currentStateForVertex2);
        DoubleTensor cov = DoubleTensor.concat(sigmaForVertex1, sigmaForVertex2).pow(2.).diag();
        DoubleTensor x = DoubleTensor.concat(proposedStateForVertex1, proposedStateForVertex2);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(mu, cov).logProb(x);
        assertThat(logProb, closeTo(expectedLogProb.sum(), 1e-14)); // sum of .log().sum() has rounding errors
    }

    @Test
    public void theLogProbAtFromIsMultivariateGaussian() {
        double logProb = proposalDistribution.logProbAtFromGivenTo(proposal);
        DoubleTensor mu = DoubleTensor.concat(proposedStateForVertex1, proposedStateForVertex2);
        DoubleTensor cov = DoubleTensor.concat(sigmaForVertex1, sigmaForVertex2).pow(2.).diag();
        DoubleTensor x = DoubleTensor.concat(currentStateForVertex1, currentStateForVertex2);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(mu, cov).logProb(x);
        assertThat(logProb, closeTo(expectedLogProb.sum(), 1e-14)); // sum of .log().sum() has rounding errors
    }

    @Test
    public void youCanAddProposalListeners() {
        ProposalListener listener1 = mock(ProposalListener.class);
        ProposalListener listener2 = mock(ProposalListener.class);
        List<ProposalListener> listeners = ImmutableList.of(listener1, listener2);
        proposalDistribution = new GaussianProposalDistribution(sigmas, listeners);
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

    @Test
    public void itThrowsIfAnEmptySigmaMapIsSpecified() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Gaussian proposal requires at least one sigma");
        new GaussianProposalDistribution(new HashMap<>());
    }

    @Test
    public void getProposalThrowsIfProposalIsMissingASigmaForAVariable() {
        CauchyVertex notInProposalDistribution = new CauchyVertex(1., 1.);
        thrown.expect(IllegalStateException.class);
        thrown.expectMessage("Gaussian proposal is missing a sigma for variable " + notInProposalDistribution);
        proposalDistribution.getProposal(ImmutableSet.of(notInProposalDistribution), KeanuRandom.getDefaultRandom());
    }

    @Test
    public void logProbThrowsIfProposalIsMissingASigmaForAVariable() {
        CauchyVertex notInProposalDistribution = new CauchyVertex(1., 1.);
        thrown.expect(IllegalStateException.class);
        thrown.expectMessage("Gaussian proposal is missing a sigma for variable " + notInProposalDistribution);
        proposalDistribution.logProb(notInProposalDistribution, DoubleTensor.scalar(1.), DoubleTensor.scalar(2.));
    }
}
