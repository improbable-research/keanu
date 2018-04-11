package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

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
        A = new ConstantDoubleVertex(2.0);
        B = new ConstantDoubleVertex(2.0);
        C = new LogVertex(A);
        D = A.multiply(B);
        E = C.plus(D);
        F = D.minus(E);
        G = new LogVertex(F);
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

    private void assertFindsAllVertices(Vertex<?> v) {
        Set<Vertex<?>> vertices = DiscoverGraph.getEntireGraph(v);
        assertTrue(vertices.size() == allVertices.size());
        assertTrue(vertices.containsAll(allVertices));
    }
}
