package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
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
import java.util.Map;
import java.util.Set;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class MultivariateGaussianProposalDistributionTest {

    public Proposal scalarProposal;
    private static final DoubleTensor scalarCurrentStateForVertex1 = DoubleTensor.create(4.2);
    private static final DoubleTensor scalarCurrentStateForVertex2 = DoubleTensor.create(42.0);

    private static final DoubleTensor scalarProposedStateForVertex1 = DoubleTensor.create(4.3);
    private static final DoubleTensor scalarProposedStateForVertex2 = DoubleTensor.create(43.0);

    private static final DoubleTensor scalarSigmaForVertex1 = DoubleTensor.create(1.);
    private static final DoubleTensor scalarSigmaForVertex2 = DoubleTensor.create(2.);

    public Proposal nonScalarProposal;
    private static final DoubleTensor nonScalarCurrentStateForVertex1 = DoubleTensor.create(4.2, 4.7);
    private static final DoubleTensor nonScalarCurrentStateForVertex2 = DoubleTensor.create(42.0, 48.0);

    private static final DoubleTensor nonScalarProposedStateForVertex1 = DoubleTensor.create(4.3, 5.8);
    private static final DoubleTensor nonScalarProposedStateForVertex2 = DoubleTensor.create(43.0, 32.0);

    private static final DoubleTensor nonScalarSigmaForVertex1 = DoubleTensor.create(1., 3.);
    private static final DoubleTensor nonScalarSigmaForVertex2 = DoubleTensor.create(2., 2.);

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Mock
    private GaussianVertex scalarVertex1;
    @Mock
    private GaussianVertex scalarVertex2;
    @Mock
    private GaussianVertex nonScalarVertex1;
    @Mock
    private GaussianVertex nonScalarVertex2;

    private MultivariateGaussianProposalDistribution scalarProposalDistribution;
    private MultivariateGaussianProposalDistribution nonScalarProposalDistribution;
    private Map<Variable, DoubleTensor> scalarSigmas;
    private Map<Variable, DoubleTensor> nonScalarSigmas;

    @Before
    public void setUpProposalDistribution() {
        scalarSigmas = ImmutableMap.of(
            scalarVertex1, scalarSigmaForVertex1,
            scalarVertex2, scalarSigmaForVertex2
        );
        nonScalarSigmas = ImmutableMap.of(
            nonScalarVertex1, nonScalarSigmaForVertex1,
            nonScalarVertex2, nonScalarSigmaForVertex2
        );
        scalarProposalDistribution = new MultivariateGaussianProposalDistribution(scalarSigmas);
        nonScalarProposalDistribution = new MultivariateGaussianProposalDistribution(nonScalarSigmas);
    }

    @Before
    public void setRandomSeed() {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Before
    public void setUpProposal() {
        when(scalarVertex1.getValue()).thenReturn(scalarCurrentStateForVertex1);
        when(scalarVertex2.getValue()).thenReturn(scalarCurrentStateForVertex2);

        when(scalarVertex1.getShape()).thenReturn(new long[] {});
        when(scalarVertex2.getShape()).thenReturn(new long[] {});

        scalarProposal = new Proposal();
        scalarProposal.setProposal(scalarVertex1, scalarProposedStateForVertex1);
        scalarProposal.setProposal(scalarVertex2, scalarProposedStateForVertex2);

        when(nonScalarVertex1.getValue()).thenReturn(nonScalarCurrentStateForVertex1);
        when(nonScalarVertex2.getValue()).thenReturn(nonScalarCurrentStateForVertex2);

        when(nonScalarVertex1.getShape()).thenReturn(new long[] {2});
        when(nonScalarVertex2.getShape()).thenReturn(new long[] {2});

        nonScalarProposal = new Proposal();
        nonScalarProposal.setProposal(nonScalarVertex1, nonScalarProposedStateForVertex1);
        nonScalarProposal.setProposal(nonScalarVertex2, nonScalarProposedStateForVertex2);
    }

    @Test
    public void theLogProbAtToIsMultivariateGaussian() {
        double logProb = scalarProposalDistribution.logProbAtToGivenFrom(scalarProposal);
        DoubleTensor mu = DoubleTensor.concat(scalarCurrentStateForVertex1, scalarCurrentStateForVertex2);
        DoubleTensor cov = DoubleTensor.concat(scalarSigmaForVertex1, scalarSigmaForVertex2).diag().pow(2);
        DoubleTensor x = DoubleTensor.concat(scalarProposedStateForVertex1, scalarProposedStateForVertex2);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(mu, cov).logProb(x);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void theLogProbAtFromIsMultivariateGaussian() {
        double logProb = scalarProposalDistribution.logProbAtFromGivenTo(scalarProposal);
        DoubleTensor mu = DoubleTensor.concat(scalarProposedStateForVertex1, scalarProposedStateForVertex2);
        DoubleTensor cov = DoubleTensor.concat(scalarSigmaForVertex1.pow(2.), scalarSigmaForVertex2.pow(2.)).diag();
        DoubleTensor x = DoubleTensor.concat(scalarCurrentStateForVertex1, scalarCurrentStateForVertex2);
        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(mu, cov).logProb(x);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void theLogProbAtToIsMultivariateGaussian_nonScalar() {
        double logProb = nonScalarProposalDistribution.logProbAtToGivenFrom(nonScalarProposal);

        DoubleTensor mu = DoubleTensor.concat(nonScalarCurrentStateForVertex1, nonScalarCurrentStateForVertex2);
        DoubleTensor cov = DoubleTensor.concat(nonScalarSigmaForVertex1, nonScalarSigmaForVertex2).pow(2.).diag();
        DoubleTensor x = DoubleTensor.concat(nonScalarProposedStateForVertex1, nonScalarProposedStateForVertex2);

        DoubleTensor expectedLogProb = MultivariateGaussian.withParameters(mu, cov).logProb(x);
        assertThat(logProb, equalTo(expectedLogProb.sum()));
    }

    @Test
    public void youCanAddProposalListeners() {
        ProposalListener listener1 = mock(ProposalListener.class);
        ProposalListener listener2 = mock(ProposalListener.class);
        List<ProposalListener> listeners = ImmutableList.of(listener1, listener2);
        scalarProposalDistribution = new MultivariateGaussianProposalDistribution(scalarSigmas, listeners);
        Set<Variable> variables = ImmutableSet.of(scalarVertex1, scalarVertex2);
        Proposal proposal = scalarProposalDistribution.getProposal(variables, KeanuRandom.getDefaultRandom());
        verify(listener1).onProposalCreated(proposal);
        verify(listener2).onProposalCreated(proposal);
        scalarProposalDistribution.onProposalRejected();
        verify(listener1).onProposalRejected(proposal);
        verify(listener2).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener1, listener2);
    }

    @Test
    public void itThrowsIfYouUseItOnADiscreteVariable() {
        thrown.expect(IllegalStateException.class);
        thrown.expectMessage("Multivariate Gaussian scalarProposal function cannot be used for discrete variable");
        PoissonVertex poisson = new PoissonVertex(1.);
        scalarProposalDistribution.getProposal(ImmutableSet.of(poisson), KeanuRandom.getDefaultRandom());
    }
}
