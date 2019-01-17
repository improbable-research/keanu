package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import lombok.AllArgsConstructor;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MetropolisHastingsStepTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom alwaysAccept;
    private KeanuRandom alwaysReject;

    @Before
    public void setup() {
        alwaysAccept = mock(KeanuRandom.class);
        when(alwaysAccept.nextDouble()).thenReturn(0.0);
        when(alwaysAccept.nextGaussian(any())).thenReturn(DoubleTensor.ZERO_SCALAR);

        alwaysReject = mock(KeanuRandom.class);
        when(alwaysReject.nextDouble()).thenReturn(1.0);
        when(alwaysReject.nextGaussian(any())).thenReturn(DoubleTensor.ZERO_SCALAR);
    }

    @Test
    public void doesCalculateCorrectLogProbAfterAcceptingStep() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(1.0);
        DoubleVertex B = A.times(2);
        DoubleVertex observedB = new GaussianVertex(B, 1);
        observedB.observe(5);

        BayesianNetwork bayesNet = new BayesianNetwork(A.getConnectedGraph());
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);
        double logProbBeforeStep = model.logProb();

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            model,
            ProposalDistribution.usePrior(),
            new RollBackOnRejection(bayesNet.getLatentVertices()),
            new LambdaSectionOptimizedLogProbCalculator(bayesNet.getLatentVertices()),
            new CascadeOnApplication(),
            alwaysAccept
        );

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            logProbBeforeStep
        );

        assertTrue(result.isAccepted());
        assertEquals(model.logProb(), result.getLogProbabilityAfterStep(), 1e-10);
    }

    @Category(Slow.class)
    @Test
    public void doesAllowCustomProposalDistribution() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(0.0);
        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(model, 1.0, alwaysAccept);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            model.logProb()
        );

        assertTrue(result.isAccepted());
        assertEquals(1.0, A.getValue(0), 1e-10);
    }

    @Test
    public void doesRejectOnImpossibleProposal() {
        DoubleVertex A = new UniformVertex(0, 1);
        A.setValue(0.5);
        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(model, -1, alwaysAccept);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            model.logProb()
        );

        assertFalse(result.isAccepted());
        assertEquals(0.5, A.getValue(0), 1e-10);
    }


    @Test
    public void doesRejectWhenRejectProbabilityIsOne() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(0.5);
        DoubleVertex B = A.times(2);
        DoubleVertex C = new GaussianVertex(B, 1);
        C.observe(5.0);
        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        MetropolisHastingsStep mhStep = stepFunctionWithConstantProposal(model, 10, alwaysReject);

        MetropolisHastingsStep.StepResult result = mhStep.step(
            Collections.singleton(A),
            model.logProb()
        );

        assertFalse(result.isAccepted());
        assertEquals(0.5, A.getValue(0), 1e-10);
    }

    private MetropolisHastingsStep stepFunctionWithConstantProposal(ProbabilisticModel model, double constant, KeanuRandom random) {
        List<Vertex> latentVertices = (List<Vertex>) model.getLatentVariables();

        return new MetropolisHastingsStep(
            model,
            constantProposal(constant),
            new RollBackOnRejection(latentVertices),
            new LambdaSectionOptimizedLogProbCalculator(latentVertices),
            new CascadeOnApplication(),
            random
        );
    }

    private ProposalDistribution constantProposal(double constant) {
        return new ConstantProposalDistribution(constant);
    }

    @AllArgsConstructor
    private static class ConstantProposalDistribution extends PriorProposalDistribution {

        private final double constant;

        @Override
        public Proposal getProposal(Set<Variable> variables, KeanuRandom random) {
            Proposal proposal = new Proposal();
            variables.forEach(variable -> proposal.setProposal(variable, DoubleTensor.scalar(constant)));
            return proposal;
        }
    }

}
