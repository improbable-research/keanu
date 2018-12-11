package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class StepsizeTest {

    private KeanuRandom random = new KeanuRandom(1);

    @Test
    public void canFindSmallStartingStepsizeForSmallSpace() {
        DoubleVertex vertex = new GaussianVertex(0, 0.05);
        List<DoubleVertex> vertices = Arrays.asList(vertex);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertex.getConnectedGraph());

        VertexId vertexId = vertex.getId();

        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(bayesianNetwork.getLatentOrObservedVertices(), vertices);
        vertex.setValue(DoubleTensor.scalar(1.));
        Map<VertexId, DoubleTensor> position = Collections.singletonMap(vertexId, vertex.getValue());
        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

        double startingStepsize = Stepsize.findStartingStepSize(
            position,
            gradient,
            Arrays.asList(vertex),
            bayesianNetwork.getLatentVertices(),
            logProbGradientCalculator,
            ProbabilityCalculator.calculateLogProbFor(vertices),
            random
        );

        double startingEpsilon = 1.0;
        Assert.assertTrue(startingStepsize < startingEpsilon);
    }

    @Test
    public void canFindLargeStartingStepsizeForLargeSpace() {
        DoubleVertex vertex = new GaussianVertex(0, 500.);
        List<DoubleVertex> vertices = Arrays.asList(vertex);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertex.getConnectedGraph());

        VertexId vertexId = vertex.getId();

        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(bayesianNetwork.getLatentOrObservedVertices(), vertices);
        Map<VertexId, DoubleTensor> position = Collections.singletonMap(vertexId, vertex.sample(random));
        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

        double startingStepsize = Stepsize.findStartingStepSize(
            position,
            gradient,
            Arrays.asList(vertex),
            bayesianNetwork.getLatentVertices(),
            logProbGradientCalculator,
            ProbabilityCalculator.calculateLogProbFor(vertices),
            random
        );

        Assert.assertTrue(startingStepsize > 64);
    }

    @Test
    public void canReduceStepsizeFromLargeInitialToSmallToExploreSmallSpace() {
        double startingStepsize = 10.;

        Stepsize tune = new Stepsize(
            startingStepsize,
            0.65,
            50
        );

        TreeBuilder mockedLessLikelyTree = mock(TreeBuilder.class);
        when(mockedLessLikelyTree.getDeltaLikelihoodOfLeapfrog()).thenAnswer(i -> -50.);
        when(mockedLessLikelyTree.getTreeSize()).thenAnswer(i -> 8.);
        double adaptedStepSizeLessLikely = tune.adaptStepSize(mockedLessLikelyTree, 1);

        Assert.assertTrue(adaptedStepSizeLessLikely < startingStepsize);

        TreeBuilder mockedLikelyTree = mock(TreeBuilder.class);
        when(mockedLikelyTree.getDeltaLikelihoodOfLeapfrog()).thenAnswer(i -> 50.);
        when(mockedLikelyTree.getTreeSize()).thenAnswer(i -> 8.);
        double adaptedStepSizeMoreLikely = tune.adaptStepSize(mockedLikelyTree, 1);

        Assert.assertTrue(adaptedStepSizeMoreLikely > startingStepsize);
    }

}
