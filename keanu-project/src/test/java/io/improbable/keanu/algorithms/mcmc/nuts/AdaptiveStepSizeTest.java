package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;
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
    public void canReduceAndIncreaseStep() {
        double startingStepSize = 0.1;

        AdaptiveStepSize tune = new AdaptiveStepSize(
            startingStepSize,
            0.65,
            6
        );

        shouldIncreaseStepSize(0.7, tune);
        shouldIncreaseStepSize(0.9, tune);
        shouldDecreaseStepSize(0.6, tune);
        shouldDecreaseStepSize(0.5, tune);
        shouldIncreaseStepSize(0.66, tune);
        shouldDecreaseStepSize(0.2, tune);

        //switches to frozen step size
        getNewStepSize(0.5, tune);

        //step size should no longer adapt
        shouldNotChangeStepSize(0.8, tune);
        shouldNotChangeStepSize(0.5, tune);
        shouldNotChangeStepSize(0.2, tune);
    }

    private void shouldNotChangeStepSize(double resultAcceptanceProb, AdaptiveStepSize tune) {

        double startStepSize = tune.getStepSize();
        double newStepSize = getNewStepSize(resultAcceptanceProb, tune);

        assertThat(newStepSize, equalTo(startStepSize));
    }

    private void shouldIncreaseStepSize(double resultAcceptanceProb, AdaptiveStepSize tune) {

        double startStepSize = tune.getStepSize();
        double newStepSize = getNewStepSize(resultAcceptanceProb, tune);

        assertThat(newStepSize, greaterThan(startStepSize));
    }

    private void shouldDecreaseStepSize(double resultAcceptanceProb, AdaptiveStepSize tune) {

        double startStepSize = tune.getStepSize();
        double newStepSize = getNewStepSize(resultAcceptanceProb, tune);

        assertThat(newStepSize, lessThan(startStepSize));
    }

    private double getNewStepSize(double resultAcceptanceProb, AdaptiveStepSize tune) {

        Tree mockedTree = mock(Tree.class);
        when(mockedTree.getSumMetropolisAcceptanceProbability()).thenAnswer(i -> resultAcceptanceProb);
        when(mockedTree.getTreeSize()).thenAnswer(i -> 1);

        return tune.adaptStepSize(mockedTree);
    }

    @Test
    public void givenAdaptStepSizeZeroThenShouldReturnInitialStepSize() {

        double initialStepSize = 0.1;
        AdaptiveStepSize tune = new AdaptiveStepSize(
            initialStepSize,
            0.65,
            0
        );

        assertEquals(initialStepSize, tune.getStepSize());

        Tree mockedTree = mock(Tree.class);
        when(mockedTree.getSumMetropolisAcceptanceProbability()).thenAnswer(i -> 0.65);
        when(mockedTree.getTreeSize()).thenAnswer(i -> 1);

        assertEquals(initialStepSize, tune.adaptStepSize(mockedTree), 1e-9);
        assertEquals(initialStepSize, tune.getStepSize(), 1e-9);
    }

}
