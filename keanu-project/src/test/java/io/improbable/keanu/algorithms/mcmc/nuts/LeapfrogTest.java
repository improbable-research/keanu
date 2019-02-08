package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LeapfrogTest {

    private static final double EPSILON = 1.0;

    private DoubleVertex vertexA = new GaussianVertex(0, 1);
    private DoubleVertex vertexB = new GaussianVertex(0, 1);

    private VariableReference aID = vertexA.getId();
    private VariableReference bID = vertexB.getId();

    private List<Vertex<DoubleTensor>> vertices = Arrays.asList(vertexA, vertexB);
    private List<VariableReference> ids = Arrays.asList(aID, bID);

    private Map<VariableReference, DoubleTensor> position = new HashMap<>();
    private Map<VariableReference, DoubleTensor> momentum = new HashMap<>();
    private Map<VariableReference, DoubleTensor> gradient = new HashMap<>();

    private ProbabilisticModelWithGradient mockedGradientCalculator;
    private Potential potential;

    @Before
    public void setupGraphForLeapfrog() {
        fillMap(position, DoubleTensor.scalar(0.0));
        fillMap(momentum, DoubleTensor.scalar(1.0));
        fillMap(gradient, DoubleTensor.scalar(2.0));
        potential = new AdaptiveQuadraticPotential(zeros(position), ones(position), 1, 100, KeanuRandom.getDefaultRandom());
    }

    @Before
    public void setupGradientMocks() {
        mockedGradientCalculator = setUpMock(1., -1.);
    }

    private ProbabilisticModelWithGradient setUpMock(double aValue, double bValue) {
        Map<VariableReference, DoubleTensor> gradient = new HashMap<>();
        gradient.put(aID, DoubleTensor.scalar(aValue));
        gradient.put(bID, DoubleTensor.scalar(bValue));

        ProbabilisticModelWithGradient mock = mock(ProbabilisticModelWithGradient.class);
        when(mock.logProbGradients(anyMap())).thenAnswer(
            invocation -> gradient
        );
        return mock;
    }

    @Test
    public void canLeapForward() {
        Leapfrog start = new Leapfrog(position, momentum, gradient, 0, potential);
        Leapfrog leap = start.step(vertices, mockedGradientCalculator, EPSILON);

        Assert.assertEquals(2.0, leap.getPosition().get(aID).scalar(), 1e-6);
        Assert.assertEquals(2.0, leap.getPosition().get(bID).scalar(), 1e-6);

        Assert.assertEquals(2.5, leap.getMomentum().get(aID).scalar(), 1e-6);
        Assert.assertEquals(1.5, leap.getMomentum().get(bID).scalar(), 1e-6);

        Assert.assertEquals(1.0, leap.getGradient().get(aID).scalar(), 1e-6);
        Assert.assertEquals(-1.0, leap.getGradient().get(bID).scalar(), 1e-6);
    }

    @Test
    public void canLeapForwardAndBackToOriginalPosition() {
        Leapfrog start = new Leapfrog(position, momentum, gradient, 0, potential);
        Leapfrog leapForward = start.step(vertices, mockedGradientCalculator, EPSILON);

        Assert.assertEquals(2.0, leapForward.getPosition().get(aID).scalar(), 1e-6);
        Assert.assertEquals(2.0, leapForward.getPosition().get(bID).scalar(), 1e-6);

        Leapfrog leapBackToStart = leapForward.step(vertices, mockedGradientCalculator, -EPSILON);

        Assert.assertEquals(0.0, leapBackToStart.getPosition().get(aID).scalar(), 1e-6);
        Assert.assertEquals(0.0, leapBackToStart.getPosition().get(bID).scalar(), 1e-6);
    }

    @Test
    public void doesIncreaseKineticEnergyWhenLogProbIncreases() {
        Leapfrog start = new Leapfrog(position, momentum, gradient, 0, potential);
        double startEnergy = start.getEnergy();
        Leapfrog leap = start.step(vertices, mockedGradientCalculator, EPSILON);
        double afterLeapEnergy = leap.getEnergy();


    }

    private void fillMap(Map<VariableReference, DoubleTensor> map, DoubleTensor value) {
        for (VariableReference id : ids) {
            map.put(id, value);
        }
    }

    private Map<VariableReference, DoubleTensor> revertDirectionOfMap(Map<VariableReference, DoubleTensor> map) {
        for (DoubleTensor tensor : map.values()) {
            tensor.timesInPlace(-1.);
        }
        return map;
    }

}