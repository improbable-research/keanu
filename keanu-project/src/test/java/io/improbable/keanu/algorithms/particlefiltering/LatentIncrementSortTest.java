package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class LatentIncrementSortTest {

    @Test
    public void simpleGraph() {
        DoubleTensorVertex muA = new ConstantDoubleTensorVertex(0.0);
        DoubleTensorVertex muB = new ConstantDoubleTensorVertex(3.0);
        DoubleTensorVertex sigma = new TensorUniformVertex(1.0, 2.0);

        DoubleTensorVertex gA = new TensorGaussianVertex(muA, sigma);
        DoubleTensorVertex gB = new TensorGaussianVertex(muB, sigma);
        DoubleTensorVertex sum = gA.plus(gB);
        DoubleTensorVertex fuzzySum = new TensorGaussianVertex(sum, sigma);

        gA.observe(0.1);
        fuzzySum.observe(1.0);

        Map<Vertex, Set<Vertex>> dependencies = LatentIncrementSort.sort(sigma.getConnectedGraph());

        assertEquals(2, dependencies.size());
        assertEquals(1, dependencies.get(gA).size());
        assertEquals(1, dependencies.get(fuzzySum).size());
        assertTrue(dependencies.get(gA).contains(sigma));
        assertTrue(dependencies.get(fuzzySum).contains(gB));
    }

    @Test
    public void moreComplexGraph() {
        DoubleTensorVertex mu = new ConstantDoubleTensorVertex(0.0);
        DoubleTensorVertex sigma1 = new TensorUniformVertex(1.0, 2.0);
        DoubleTensorVertex g1 = new TensorGaussianVertex(mu, sigma1);
        g1.observe(0.0);

        DoubleTensorVertex sigma2 = new TensorUniformVertex(1.0, 2.0);
        DoubleTensorVertex g2 = new TensorGaussianVertex(g1, sigma2);

        DoubleTensorVertex sigma3 = new TensorUniformVertex(1.0, 2.0);
        DoubleTensorVertex g3 = new TensorGaussianVertex(g2, sigma3);
        g3.observe(0.0);

        DoubleTensorVertex g4 = new TensorGaussianVertex(g3, sigma3);
        DoubleTensorVertex sigma4 = new TensorUniformVertex(1.0, 2.0);
        DoubleTensorVertex g5 = new TensorGaussianVertex(g4, sigma4);

        DoubleTensorVertex sigma5 = new TensorUniformVertex(1.0, 2.0);
        DoubleTensorVertex g6 = new TensorGaussianVertex(g5, sigma5);
        g6.observe(0.0);

        Map<Vertex, Set<Vertex>> dependencies = LatentIncrementSort.sort(mu.getConnectedGraph());

        assertEquals(3, dependencies.size());
        assertEquals(1, dependencies.get(g1).size());
        assertTrue(dependencies.get(g1).contains(sigma1));

        assertEquals(3, dependencies.get(g3).size());
        assertTrue(dependencies.get(g3).contains(g2) &&
            dependencies.get(g3).contains(sigma2) &&
            dependencies.get(g3).contains(sigma3));

        assertEquals(4, dependencies.get(g6).size());
        assertTrue(dependencies.get(g6).contains(g4) &&
            dependencies.get(g6).contains(sigma4) &&
            dependencies.get(g6).contains(g5) &&
            dependencies.get(g6).contains(sigma5));

        List<Vertex<?>> expectedOrder = Arrays.asList(g1, g3, g6);
        int idx = 0;
        for (Map.Entry<Vertex, Set<Vertex>> entry : dependencies.entrySet()) {
            assertEquals(entry.getKey(), expectedOrder.get(idx));
            idx++;
        }
    }
}
