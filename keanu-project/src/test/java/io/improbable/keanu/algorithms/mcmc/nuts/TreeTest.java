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

import static io.improbable.keanu.algorithms.mcmc.nuts.LeapfrogTest.leapfrogAt;
import static java.util.Collections.singletonList;
import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.not;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class TreeTest {

    GaussianVertex vertex;
    Leapfrog start;
    KeanuProbabilisticModelWithGradient gradientCalculator;

    @Before
    public void setup() {
        vertex = new GaussianVertex(0, 1);

        gradientCalculator = new KeanuProbabilisticModelWithGradient(
            new BayesianNetwork(vertex.getConnectedGraph())
        );

        start = leapfrogAt(vertex, 0.0, 0.5);
    }

    @Test
    public void treeSizeAndHeightGrows() {

        Tree tree = new Tree(
            start,
            null,
            1000,
            gradientCalculator,
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        tree.grow(1, 1e-6);
        tree.grow(-1, 1e-6);
        tree.grow(-1, 1e-6);

        assertEquals(tree.getTreeHeight(), 3);
        assertEquals(tree.getTreeSize(), (int) Math.pow(2, tree.getTreeHeight()));
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
            singletonList(vertex),
            KeanuRandom.getDefaultRandom()
        );

        tree.grow(1, 1e-6);
        verify(mockModel, times(tree.getTreeSize() - 1)).logProbGradients(anyMap());

        tree.grow(-1, 1e-6);
        verify(mockModel, times(tree.getTreeSize() - 1)).logProbGradients(anyMap());

        tree.grow(-1, 1e-6);
        verify(mockModel, times(tree.getTreeSize() - 1)).logProbGradients(anyMap());

        tree.grow(1, 1e-6);
        verify(mockModel, times(tree.getTreeSize() - 1)).logProbGradients(anyMap());
    }

    @Test
    public void treeGrowthMovesForwardsAndBackwards() {

        Tree tree = new Tree(
            start,
            null,
            1000,
            gradientCalculator,
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
    public void canLogSumExp() {

        double a = -0.25;
        double b = 0.5;
        double expected = Math.log(Math.exp(a) + Math.exp(b));
        double actual = Tree.logSumExp(a, b);

        assertEquals(expected, actual, 1e-6);
    }

}
