package io.improbable.keanu.algorithms.particlefiltering;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class LatentIncrementSortTest {

    @Test
    public void simpleGraph() {
        DoubleVertex muA = ConstantVertex.of(0.0);
        DoubleVertex muB = ConstantVertex.of(3.0);
        DoubleVertex sigma = VertexOfType.uniform(1.0, 2.0);

        DoubleVertex gA = VertexOfType.gaussian(muA, sigma);
        DoubleVertex gB = VertexOfType.gaussian(muB, sigma);
        DoubleVertex sum = gA.plus(gB);
        DoubleVertex fuzzySum = VertexOfType.gaussian(sum, sigma);

        gA.observe(DoubleTensor.scalar(0.1));
        fuzzySum.observe(DoubleTensor.scalar(1.0));

        Map<Vertex, Set<Vertex>> dependencies = LatentIncrementSort.sort(sigma.getConnectedGraph());

        assertEquals(2, dependencies.size());
        assertEquals(1, dependencies.get(gA).size());
        assertEquals(1, dependencies.get(fuzzySum).size());
        assertTrue(dependencies.get(gA).contains(sigma));
        assertTrue(dependencies.get(fuzzySum).contains(gB));
    }

    @Test
    public void moreComplexGraph() {
        DoubleVertex mu = ConstantVertex.of(0.0);
        DoubleVertex sigma1 = VertexOfType.uniform(1.0, 2.0);
        DoubleVertex g1 = VertexOfType.gaussian(mu, sigma1);
        g1.observe(DoubleTensor.scalar(0.0));

        DoubleVertex sigma2 = VertexOfType.uniform(1.0, 2.0);
        DoubleVertex g2 = VertexOfType.gaussian(g1, sigma2);

        DoubleVertex sigma3 = VertexOfType.uniform(1.0, 2.0);
        DoubleVertex g3 = VertexOfType.gaussian(g2, sigma3);
        g3.observe(DoubleTensor.scalar(0.0));

        DoubleVertex g4 = VertexOfType.gaussian(g3, sigma3);
        DoubleVertex sigma4 = VertexOfType.uniform(1.0, 2.0);
        DoubleVertex g5 = VertexOfType.gaussian(g4, sigma4);

        DoubleVertex sigma5 = VertexOfType.uniform(1.0, 2.0);
        DoubleVertex g6 = VertexOfType.gaussian(g5, sigma5);
        g6.observe(DoubleTensor.scalar(0.0));

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
