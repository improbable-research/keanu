package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.LeapfrogIntegratorTest.leapfrogAt;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static io.improbable.keanu.tensor.dbl.DoubleTensor.scalar;
import static java.util.Collections.singletonList;
import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.not;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class TreeTest {

    private GaussianVertex vertex;
    private LeapfrogState start;
    private KeanuProbabilisticModelWithGradient gradientCalculator;
    private LeapfrogIntegrator leapfrogIntegrator;

    @Before
    public void setup() {
        vertex = new GaussianVertex(0, 1);

        gradientCalculator = new KeanuProbabilisticModelWithGradient(
            new BayesianNetwork(vertex.getConnectedGraph())
        );

        Map<VariableReference, DoubleTensor> p = ImmutableMap.of(vertex.getId(), scalar(0));

        Potential potential = new AdaptiveQuadraticPotential(zeros(p), ones(p), 1, 100, 101, KeanuRandom.getDefaultRandom());

        start = leapfrogAt(vertex, 0.0, 0.5, potential);

        leapfrogIntegrator = new LeapfrogIntegrator(potential);
    }

    @Test
    public void treeSizeAndHeightGrows() {

        Tree tree = new Tree(
            start,
            null,
            1000,
            gradientCalculator,
            leapfrogIntegrator,
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        tree.grow(1, 1e-6);
        tree.grow(-1, 1e-6);
        tree.grow(-1, 1e-6);

        assertEquals(3, tree.getTreeHeight());
        assertEquals((int) Math.pow(2, tree.getTreeHeight()) - 1, tree.getTreeSize());
    }

    @Test
    public void treeGrowthEvaluatesOneLogProbPerStep() {

        ProbabilisticModelWithGradient mockModel = mock(ProbabilisticModelWithGradient.class);

        Map<VariableReference, DoubleTensor> mockGradient = ImmutableMap.of(vertex.getId(), DoubleTensor.scalar(0.0));

        when(mockModel.logProbGradients(anyMap()))
            .thenReturn(mockGradient);

        Tree tree = new Tree(
            start,
            null,
            1000,
            mockModel,
            leapfrogIntegrator,
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        tree.grow(1, 1e-6);
        verify(mockModel, times(tree.getTreeSize())).logProbGradients(anyMap());

        tree.grow(-1, 1e-6);
        verify(mockModel, times(tree.getTreeSize())).logProbGradients(anyMap());

        tree.grow(-1, 1e-6);
        verify(mockModel, times(tree.getTreeSize())).logProbGradients(anyMap());

        tree.grow(1, 1e-6);
        verify(mockModel, times(tree.getTreeSize())).logProbGradients(anyMap());
    }

    @Test
    public void treeGrowthMovesForwardsAndBackwards() {

        Tree tree = new Tree(
            start,
            null,
            1000,
            gradientCalculator,
            leapfrogIntegrator,
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        assertMovesInCorrectDirection(tree, 1);
        assertMovesInCorrectDirection(tree, -1);
        assertMovesInCorrectDirection(tree, 1);
        assertMovesInCorrectDirection(tree, 1);
        assertMovesInCorrectDirection(tree, -1);
    }

    private void assertMovesInCorrectDirection(Tree tree, int direction) {

        double forwardPositionBefore = tree.getForward().getPosition().get(vertex.getId()).scalar();
        double backwardPositionBefore = tree.getBackward().getPosition().get(vertex.getId()).scalar();

        tree.grow(direction, 1e-6);

        double forwardPositionAfter = tree.getForward().getPosition().get(vertex.getId()).scalar();
        double backwardPositionAfter = tree.getBackward().getPosition().get(vertex.getId()).scalar();

        if (direction == 1) {
            assertThat(backwardPositionBefore, closeTo(backwardPositionAfter, 1e-8));
            assertThat(forwardPositionBefore, not(closeTo(forwardPositionAfter, 1e-8)));
        } else {
            assertThat(backwardPositionBefore, not(closeTo(backwardPositionAfter, 1e-8)));
            assertThat(forwardPositionBefore, closeTo(forwardPositionAfter, 1e-8));
        }
    }

    @Test
    public void doesSumMomentumAndMetropolisAcceptanceProbAndLogSumWeight() {

        LeapfrogIntegrator leapfrogIntegrator = mock(LeapfrogIntegrator.class);

        double startEnergy = start.getEnergy();
        double energyChange = -2.0;

        LeapfrogState constantLeapfrogState = new LeapfrogState(
            start.getPosition(),
            start.getMomentum(),
            start.getVelocity(),
            start.getGradient(),
            start.getKineticEnergy(),
            start.getLogProb() + energyChange,
            startEnergy + energyChange
        );

        when(leapfrogIntegrator.step(any(), any(), anyDouble()))
            .thenReturn(constantLeapfrogState);

        Tree tree = new Tree(
            start,
            null,
            1000,
            gradientCalculator,
            leapfrogIntegrator,
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        tree.grow(1, 1e-3);
        tree.grow(1, 1e-3);
        tree.grow(1, 1e-3);

        double expectedSumMetropolisAcceptanceProb = tree.getTreeSize();
        assertThat(tree.getSumMetropolisAcceptanceProbability(), closeTo(expectedSumMetropolisAcceptanceProb, 1e-6));

        double expectedLogSumWeight = 0;
        for (int i = 0; i < tree.getTreeSize(); i++) {
            expectedLogSumWeight = Tree.logSumExp(expectedLogSumWeight, -energyChange);
        }
        assertThat(tree.getLogSumWeight(), closeTo(expectedLogSumWeight, 1e-6));

        double expectedSumMomentum = (tree.getTreeSize() + 1) * start.getMomentum().get(vertex.getReference()).scalar();
        assertThat(expectedSumMomentum, closeTo(tree.getSumMomentum().get(vertex.getReference()).scalar(), 1e-6));
    }

    @Test
    public void canLogSumExp() {

        double a = -0.25;
        double b = 0.5;
        double expected = Math.log(Math.exp(a) + Math.exp(b));
        double actual = Tree.logSumExp(a, b);

        assertEquals(expected, actual, 1e-6);
    }

}
