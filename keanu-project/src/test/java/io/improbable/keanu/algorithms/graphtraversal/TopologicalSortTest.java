package io.improbable.keanu.algorithms.graphtraversal;

import static org.junit.Assert.assertEquals;

import static junit.framework.TestCase.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class TopologicalSortTest {

    @Test
    public void sortsSimpleLinearGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex C = VertexOfType.gaussian(B, ConstantVertex.of(1.0));

        List<? extends Vertex> sorted = TopologicalSort.sort(Arrays.asList(C, B, A));
        List<Vertex<?>> expected = Arrays.asList(A, B, C);

        assertExactOrder(expected, sorted);
    }

    @Test
    public void sortsSimpleDiamondGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex C = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex D = VertexOfType.gaussian(B, C);

        List<? extends Vertex> sorted = TopologicalSort.sort(Arrays.asList(D, C, B, A));

        assertEquals(4, sorted.size());
        assertEquals(sorted.get(0), A);
        assertEquals(sorted.get(3), D);
    }

    @Test
    public void sortsDoubleDiamondGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex C = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex D = VertexOfType.gaussian(B, C);
        DoubleVertex E = VertexOfType.gaussian(D, ConstantVertex.of(1.0));
        DoubleVertex F = VertexOfType.gaussian(D, ConstantVertex.of(1.0));
        DoubleVertex G = VertexOfType.gaussian(E, F);

        List<Vertex> vertices = Arrays.asList(F, B, G, C, E, A);
        List<? extends Vertex> sorted = TopologicalSort.sort(vertices);

        assertEquals(sorted.size(), vertices.size());
        assertEquals(sorted.get(0), A);
        assertEquals(sorted.get(5), G);
    }

    @Test
    public void sortsComplexGraph() {
        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex C = VertexOfType.gaussian(A, B);
        DoubleVertex D = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex F = VertexOfType.gaussian(C, D);
        DoubleVertex E = VertexOfType.gaussian(C, F);
        DoubleVertex H = VertexOfType.gaussian(E, F);
        DoubleVertex G = VertexOfType.gaussian(E, ConstantVertex.of(1.0));
        DoubleVertex I = VertexOfType.gaussian(H, G);
        DoubleVertex J = VertexOfType.gaussian(H, ConstantVertex.of(1.0));
        DoubleVertex K = VertexOfType.gaussian(H, J);
        DoubleVertex M = VertexOfType.gaussian(J, ConstantVertex.of(1.0));
        DoubleVertex L = VertexOfType.gaussian(K, H);

        List<Vertex> vertices = Arrays.asList(H, B, L, I, F, E, A, G, D, J, K, M, C);
        List<? extends Vertex> sorted = TopologicalSort.sort(vertices);

        assertEquals(sorted.size(), vertices.size());
        assertTrue(sorted.indexOf(A) < sorted.indexOf(C));
        assertTrue(sorted.indexOf(B) < sorted.indexOf(C));
        assertTrue(sorted.indexOf(E) < sorted.indexOf(M));
        assertTrue(sorted.indexOf(F) < sorted.indexOf(E));
        assertTrue(sorted.indexOf(H) < sorted.indexOf(I));
        assertTrue(sorted.indexOf(K) < sorted.indexOf(L));
    }

    private void assertExactOrder(List<? extends Vertex<?>> expected, List<? extends Vertex> actual) {
        assertEquals(actual.size(), expected.size());
        for (int i = 0; i < actual.size(); i++) {
            assertEquals(actual.get(i), expected.get(i));
        }
    }
}
