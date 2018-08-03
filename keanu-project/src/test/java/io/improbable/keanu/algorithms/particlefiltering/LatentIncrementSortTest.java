package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.ConstantVertex;
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
        DoubleVertex muA = ConstantVertex.of(0.0);
        DoubleVertex muB = ConstantVertex.of(3.0);
        DoubleVertex sigma = new UniformVertex(1.0, 2.0);

        DoubleVertex gA = new GaussianVertex(muA, sigma);
        DoubleVertex gB = new GaussianVertex(muB, sigma);
        DoubleVertex sum = gA.plus(gB);
        DoubleVertex fuzzySum = new GaussianVertex(sum, sigma);

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
        DoubleVertex mu = ConstantVertex.of(0.0);
        DoubleVertex sigma1 = new UniformVertex(1.0, 2.0);
        DoubleVertex g1 = new GaussianVertex(mu, sigma1);
        g1.observe(0.0);

        DoubleVertex sigma2 = new UniformVertex(1.0, 2.0);
        DoubleVertex g2 = new GaussianVertex(g1, sigma2);

        DoubleVertex sigma3 = new UniformVertex(1.0, 2.0);
        DoubleVertex g3 = new GaussianVertex(g2, sigma3);
        g3.observe(0.0);

        DoubleVertex g4 = new GaussianVertex(g3, sigma3);
        DoubleVertex sigma4 = new UniformVertex(1.0, 2.0);
        DoubleVertex g5 = new GaussianVertex(g4, sigma4);

        DoubleVertex sigma5 = new UniformVertex(1.0, 2.0);
        DoubleVertex g6 = new GaussianVertex(g5, sigma5);
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
