package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class StepsizeTest {

    private KeanuRandom random = new KeanuRandom(1);

    @Test
    public void canFindSmallStartingStepsizeForSmallSpace() {
        DoubleVertex vertex = new GaussianVertex(0, 0.05);
        double startingEpsilon = 1.0;
        double stepsize = calculateStepsize(vertex, 1.);
        assertThat(stepsize, Matchers.lessThan(startingEpsilon));
    }

    @Test
    public void canFindLargeStartingStepsizeForLargeSpace() {
        DoubleVertex vertex = new GaussianVertex(0, 500.);
        double stepsize = calculateStepsize(vertex, 1.);
        assertThat(stepsize, Matchers.greaterThan(64.));
    }

    private double calculateStepsize(DoubleVertex vertex, double startingValue) {
        List<DoubleVertex> vertices = Arrays.asList(vertex);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertex.getConnectedGraph());
        KeanuProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(bayesianNetwork);

        VertexId vertexId = vertex.getId();

        vertex.setValue(DoubleTensor.scalar(startingValue));
        Map<VariableReference, DoubleTensor> position = Collections.singletonMap(vertexId, vertex.getValue());
        Map<? extends VariableReference, DoubleTensor> gradient = model.logProbGradients();

        return Stepsize.findStartingStepSize(
            position,
            gradient,
            Collections.singletonList(vertex),
            model,
            ProbabilityCalculator.calculateLogProbFor(vertices),
            random
        );
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
        when(mockedLessLikelyTree.getTreeSize()).thenAnswer(i -> 8.);
        double adaptedStepSizeLessLikely = tune.adaptStepSize(mockedLessLikelyTree, 1);

        Assert.assertTrue(adaptedStepSizeLessLikely < startingStepsize);
        assertThat(adaptedStepSizeLessLikely, Matchers.lessThan(startingStepsize));

        Tree mockedLikelyTree = mock(Tree.class);
        when(mockedLikelyTree.getDeltaLikelihoodOfLeapfrog()).thenAnswer(i -> 50.);
        when(mockedLikelyTree.getTreeSize()).thenAnswer(i -> 8.);
        double adaptedStepSizeMoreLikely = tune.adaptStepSize(mockedLikelyTree, 1);

        assertThat(adaptedStepSizeMoreLikely, Matchers.greaterThan(startingStepsize));
    }

}
