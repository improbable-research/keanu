package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class AdaptiveStepSizeTest {

    @Test
    public void canUseSimpleHeuristicForInitialStepSize() {

        double stepScale = 0.25;
        double startingStepSizeSimple = AdaptiveStepSize.findStartingStepSizeSimple(stepScale, ImmutableList.of(
            ConstantVertex.of(DoubleTensor.arange(0, 5)),
            ConstantVertex.of(DoubleTensor.arange(0, 5))
        ));

        double expected = stepScale / (Math.pow(10, 0.25));
        assertEquals(expected, startingStepSizeSimple, 1e-6);
    }

    @Test
    public void canReduceStepsizeFromLargeInitialToSmallToExploreSmallSpace() {
        double startingStepSize = 10.;

        AdaptiveStepSize tune = new AdaptiveStepSize(
            startingStepSize,
            0.65,
            50
        );

        Tree mockedLessLikelyTree = mock(Tree.class);
        when(mockedLessLikelyTree.getSumMetropolisAcceptanceProbability()).thenAnswer(i -> -50.);
        when(mockedLessLikelyTree.getTreeSize()).thenAnswer(i -> 8);
        double adaptedStepSizeLessLikely = tune.adaptStepSize(mockedLessLikelyTree, 1);

        Assert.assertTrue(adaptedStepSizeLessLikely < startingStepSize);
        assertThat(adaptedStepSizeLessLikely, Matchers.lessThan(startingStepSize));

        Tree mockedLikelyTree = mock(Tree.class);
        when(mockedLikelyTree.getSumMetropolisAcceptanceProbability()).thenAnswer(i -> 50.);
        when(mockedLikelyTree.getTreeSize()).thenAnswer(i -> 8);
        double adaptedStepSizeMoreLikely = tune.adaptStepSize(mockedLikelyTree, 1);

        assertThat(adaptedStepSizeMoreLikely, Matchers.greaterThan(startingStepSize));
    }

}
