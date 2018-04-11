package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class TopologicalSortTest {

    @Test
    public void sortsSimpleLinearGraph() {

        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));
        DoubleVertex B = new GaussianVertex(A, new ConstantDoubleVertex(1.0));
        DoubleVertex C = new GaussianVertex(B, new ConstantDoubleVertex(1.0));

        List<? extends Vertex<?>> sorted = TopologicalSort.sort(Arrays.asList(C, B, A));
        List<Vertex<?>> expected = Arrays.asList(A, B, C);

        assertExactOrder(expected, sorted);
    }

    @Test
    public void sortsSimpleDiamondGraph() {

        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));

        DoubleVertex B = new GaussianVertex(A, new ConstantDoubleVertex(1.0));
        DoubleVertex C = new GaussianVertex(A, new ConstantDoubleVertex(1.0));

        DoubleVertex D = new GaussianVertex(B, C);

        List<? extends Vertex<?>> sorted = TopologicalSort.sort(Arrays.asList(D, C, B, A));

        assertTrue(sorted.size() == 4);
        assertTrue(sorted.get(0).equals(A));
        assertTrue(sorted.get(3).equals(D));
    }

    @Test
    public void sortsDoubleDiamondGraph() {

        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));
        DoubleVertex B = new GaussianVertex(A, new ConstantDoubleVertex(1.0));
        DoubleVertex C = new GaussianVertex(A, new ConstantDoubleVertex(1.0));
        DoubleVertex D = new GaussianVertex(B, C);
        DoubleVertex E = new GaussianVertex(D, new ConstantDoubleVertex(1.0));
        DoubleVertex F = new GaussianVertex(D, new ConstantDoubleVertex(1.0));
        DoubleVertex G = new GaussianVertex(E, F);

        List<Vertex<?>> vertices = Arrays.asList(F, B, G, C, E, A);
        List<? extends Vertex<?>> sorted = TopologicalSort.sort(vertices);

        assertTrue(sorted.size() == vertices.size());
        assertTrue(sorted.get(0).equals(A));
        assertTrue(sorted.get(5).equals(G));
    }

    @Test
    public void sortsComplexGraph() {
        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));
        DoubleVertex B = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));
        DoubleVertex C = new GaussianVertex(A, B);
        DoubleVertex D = new GaussianVertex(new ConstantDoubleVertex(5.0), new ConstantDoubleVertex(1.0));
        DoubleVertex F = new GaussianVertex(C, D);
        DoubleVertex E = new GaussianVertex(C, F);
        DoubleVertex H = new GaussianVertex(E, F);
        DoubleVertex G = new GaussianVertex(E, new ConstantDoubleVertex(1.0));
        DoubleVertex I = new GaussianVertex(H, G);
        DoubleVertex J = new GaussianVertex(H, new ConstantDoubleVertex(1.0));
        DoubleVertex K = new GaussianVertex(H, J);
        DoubleVertex M = new GaussianVertex(J, new ConstantDoubleVertex(1.0));
        DoubleVertex L = new GaussianVertex(K, H);

        List<Vertex<?>> vertices = Arrays.asList(H, B, L, I, F, E, A, G, D, J, K, M, C);
        List<? extends Vertex<?>> sorted = TopologicalSort.sort(vertices);

        assertTrue(sorted.size() == vertices.size());
        assertTrue(sorted.get(3).equals(C));
        assertTrue(sorted.get(4).equals(F));
        assertTrue(sorted.get(5).equals(E));
    }

    private void assertExactOrder(List<? extends Vertex<?>> expected, List<? extends Vertex<?>> actual) {
        assertTrue(actual.size() == expected.size());
        for (int i = 0; i < actual.size(); i++) {
            assertTrue(actual.get(i).equals(expected.get(i)));
        }
    }
}
