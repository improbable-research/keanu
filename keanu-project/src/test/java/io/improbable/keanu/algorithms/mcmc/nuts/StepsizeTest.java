package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class StepsizeTest {

    private KeanuRandom random = new KeanuRandom(1);

//    @Test
//    public void canFindSmallStartingStepsizeForSmallSpace() {
//        DoubleVertex vertex = new GaussianVertex(0, 0.05);
//        double startingEpsilon = 1.0;
//        double stepsize = calculateStepSize(vertex, 1.);
//        assertThat(stepsize, Matchers.lessThan(startingEpsilon));
//    }
//
//    @Test
//    public void canFindLargeStartingStepsizeForLargeSpace() {
//        DoubleVertex vertex = new GaussianVertex(0, 500.);
//        double stepsize = calculateStepSize(vertex, 1.);
//        assertThat(stepsize, Matchers.greaterThan(64.));
//    }
//
//    private double calculateStepSize(DoubleVertex vertex, double startingValue) {
//        List<DoubleVertex> vertices = Arrays.asList(vertex);
//        KeanuProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(vertex.getConnectedGraph());
//
//        VertexId vertexId = vertex.getId();
//
//        vertex.setValue(DoubleTensor.scalar(startingValue));
//        Map<VariableReference, DoubleTensor> position = Collections.singletonMap(vertexId, vertex.getValue());
//        Map<? extends VariableReference, DoubleTensor> gradient = model.logProbGradients();
//
//        return Stepsize.findStartingStepSize(
//            position,
//            gradient,
//            Collections.singletonList(vertex),
//            model,
//            ProbabilityCalculator.calculateLogProbFor(vertices),
//            random
//        );
//    }

    @Test
    public void canUseSimpleHeuristicForInitialStepSize() {

        double startingStepSizeSimple = Stepsize.findStartingStepSizeSimple(0.25, ImmutableList.of(
            ConstantVertex.of(DoubleTensor.arange(0, 5)),
            ConstantVertex.of(DoubleTensor.arange(0, 5))
        ));

        double expected = 0.25 / (Math.pow(10, 0.25));
        assertEquals(expected, startingStepSizeSimple, 1e-6);
    }

    @Test
    public void canReduceStepsizeFromLargeInitialToSmallToExploreSmallSpace() {
        double startingStepsize = 10.;

        Stepsize tune = new Stepsize(
            startingStepsize,
            0.65,
            50
        );

        Tree mockedLessLikelyTree = mock(Tree.class);
        when(mockedLessLikelyTree.getDeltaLikelihoodOfLeapfrog()).thenAnswer(i -> -50.);
        when(mockedLessLikelyTree.getTreeSize()).thenAnswer(i -> 8);
        double adaptedStepSizeLessLikely = tune.adaptStepSize(mockedLessLikelyTree, 1);

        Assert.assertTrue(adaptedStepSizeLessLikely < startingStepsize);
        assertThat(adaptedStepSizeLessLikely, Matchers.lessThan(startingStepsize));

        Tree mockedLikelyTree = mock(Tree.class);
        when(mockedLikelyTree.getDeltaLikelihoodOfLeapfrog()).thenAnswer(i -> 50.);
        when(mockedLikelyTree.getTreeSize()).thenAnswer(i -> 8);
        double adaptedStepSizeMoreLikely = tune.adaptStepSize(mockedLikelyTree, 1);

        assertThat(adaptedStepSizeMoreLikely, Matchers.greaterThan(startingStepsize));
    }

}
