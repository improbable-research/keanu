package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.ConstantVertexFactory;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DiscoverGraphTest {

    DoubleVertex A;
    DoubleVertex B;
    DoubleVertex C;
    DoubleVertex D;
    DoubleVertex E;
    DoubleVertex F;
    DoubleVertex G;
    List<DoubleVertex> allVertices;

    @Before
    public void setup() {
        A = ConstantVertexFactory.of(2.0);
        B = ConstantVertexFactory.of(2.0);
        C = A.log();
        D = A.multiply(B);
        E = C.plus(D);
        F = D.minus(E);
        G = F.log();
        allVertices = Arrays.asList(A, B, C, D, E, F, G);
    }

    @Test
    public void getsCompleteGraphFromCenter() {
        assertFindsAllVertices(D);
    }

    @Test
    public void getsCompleteGraphFromTop() {
        assertFindsAllVertices(A);
    }

    @Test
    public void getsCompleteGraphFromEverywhere() {
        for (Vertex<?> v : allVertices) {
            assertFindsAllVertices(v);
        }
    }

    @Test
    public void findsVeryLongGraph() {

        DoubleVertex start = new GaussianVertex(0, 1);

        DoubleVertex end = start;

        int links = 15000;
        for (int i = 0; i < links; i++) {
            DoubleVertex left = end.abs();
            DoubleVertex right = end.abs();
            end = left.plus(right);
        }

        Set<Vertex> connectedGraph = end.getConnectedGraph();

        int expectedSize = 3 + 3 * links;

        assertEquals(expectedSize, connectedGraph.size());
    }

    private void assertFindsAllVertices(Vertex<?> v) {
        Set<Vertex> vertices = DiscoverGraph.getEntireGraph(v);
        assertEquals(vertices.size(), allVertices.size());
        assertTrue(vertices.containsAll(allVertices));
    }
}
