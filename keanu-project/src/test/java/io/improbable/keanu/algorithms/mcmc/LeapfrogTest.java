package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LeapfrogTest {

    private DoubleVertex A;
    private DoubleVertex B;

    private VertexId aID;
    private VertexId bID;

    private List<Vertex<DoubleTensor>> vertices;
    private List<VertexId> ids;

    private Map<VertexId, DoubleTensor> position;
    private Map<VertexId, DoubleTensor> momentum;
    private Map<VertexId, DoubleTensor> gradient;

    private double epsilon;

    private LogProbGradientCalculator mockedGradientCalculator;
    private LogProbGradientCalculator mockedReverseGradientCalculator;

    @Before
    public void setupGraphForLeapfrog() {
        A = new GaussianVertex(0, 1);
        B = new GaussianVertex(0, 1);

        vertices = Arrays.asList(A, B);

        aID = A.getId();
        bID = B.getId();

        ids = Arrays.asList(aID, bID);

        position = new HashMap<>();
        momentum = new HashMap<>();
        gradient = new HashMap<>();

        fillMap(position, DoubleTensor.scalar(0.0));
        fillMap(momentum, DoubleTensor.scalar(1.0));
        fillMap(gradient, DoubleTensor.scalar(2.0));

        epsilon = 1.0;

        Map<VertexId, DoubleTensor> mockedGradient = new HashMap<>();
        mockedGradient.put(aID, DoubleTensor.scalar(1.0));
        mockedGradient.put(bID, DoubleTensor.scalar(-1.0));

        mockedGradientCalculator = mock(LogProbGradientCalculator.class);
        when(mockedGradientCalculator.getJointLogProbGradientWrtLatents()).thenAnswer(
            invocation -> mockedGradient
        );

        Map<VertexId, DoubleTensor> mockedReverseGradient = new HashMap<>();
        mockedReverseGradient.put(aID, DoubleTensor.scalar(-1.0));
        mockedReverseGradient.put(bID, DoubleTensor.scalar(1.0));

        mockedReverseGradientCalculator = mock(LogProbGradientCalculator.class);
        when(mockedReverseGradientCalculator.getJointLogProbGradientWrtLatents()).thenAnswer(
            invocation -> mockedReverseGradient
        );
    }

    @Test
    public void canLeapForward() {
        Leapfrog start = new Leapfrog(position, momentum, gradient);
        Leapfrog leap = start.step(vertices, mockedGradientCalculator, epsilon);

        Assert.assertEquals(1.0, leap.position.get(aID).scalar(), 1e-6);
        Assert.assertEquals(1.0, leap.position.get(bID).scalar(), 1e-6);

        Assert.assertEquals(2.5, leap.momentum.get(aID).scalar(), 1e-6);
        Assert.assertEquals(1.5, leap.momentum.get(bID).scalar(), 1e-6);

        Assert.assertEquals(1.0, leap.gradient.get(aID).scalar(), 1e-6);
        Assert.assertEquals(-1.0, leap.gradient.get(bID).scalar(), 1e-6);
    }

    @Test
    public void canLeapForwardAndBackToOriginalPosition() {
        Leapfrog start = new Leapfrog(position, momentum, gradient);
        Leapfrog leapForward = start.step(vertices, mockedGradientCalculator, epsilon);

        Map<VertexId, DoubleTensor> momentum = new HashMap<>(leapForward.momentum);

        fillMap(leapForward.momentum, DoubleTensor.scalar(-1.0));
        fillMap(leapForward.gradient, DoubleTensor.scalar(-2.0));

        Leapfrog leapBackToStart = leapForward.step(vertices, mockedReverseGradientCalculator, epsilon);

        assertMapsAreEqual(start.position, leapBackToStart.position);
        assertMapsAreEqual(momentum, revertDirectionOfMap(leapBackToStart.momentum));
    }

    private void fillMap(Map<VertexId, DoubleTensor> map, DoubleTensor value) {
        for (VertexId id : ids) {
            map.put(id, value);
        }
    }

    private void assertMapsAreEqual(Map<VertexId, DoubleTensor> one, Map<VertexId, DoubleTensor> two) {
        for (VertexId id : one.keySet()) {
            Assert.assertEquals(one.get(id).scalar(), two.get(id).scalar(), 1e-6);
        }
    }

    private Map<VertexId, DoubleTensor> revertDirectionOfMap(Map<VertexId, DoubleTensor> map) {
        for (VertexId id : map.keySet()) {
            map.put(id, map.get(id).times(-1.));
        }
        return map;
    }

}