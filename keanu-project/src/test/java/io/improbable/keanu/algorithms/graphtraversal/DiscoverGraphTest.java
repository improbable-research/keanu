package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.TensorLogVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DiscoverGraphTest {

    DoubleTensorVertex A;
    DoubleTensorVertex B;
    DoubleTensorVertex C;
    DoubleTensorVertex D;
    DoubleTensorVertex E;
    DoubleTensorVertex F;
    DoubleTensorVertex G;
    List<DoubleTensorVertex> allVertices;

    @Before
    public void setup() {
        A = new ConstantDoubleTensorVertex(2.0);
        B = new ConstantDoubleTensorVertex(2.0);
        C = new TensorLogVertex(A);
        D = A.multiply(B);
        E = C.plus(D);
        F = D.minus(E);
        G = new TensorLogVertex(F);
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

        DoubleTensorVertex start = new TensorGaussianVertex(0, 1);

        DoubleTensorVertex end = start;

        int links = 15000;
        for (int i = 0; i < links; i++) {
            DoubleTensorVertex left = end.abs();
            DoubleTensorVertex right = end.abs();
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
