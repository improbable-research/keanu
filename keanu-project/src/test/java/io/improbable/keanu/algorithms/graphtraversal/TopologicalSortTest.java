package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TopologicalSortTest {

    @Test
    public void sortsSimpleLinearGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(B, 1.0);

        List<? extends Vertex> sorted = TopologicalSort.sort(Arrays.asList(C, B, A));
        List<Vertex<?>> expected = Arrays.asList(A, B, C);

        assertExactOrder(expected, sorted);
    }

    @Test
    public void sortsSimpleDiamondGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex D = new TensorGaussianVertex(B, C);

        List<? extends Vertex> sorted = TopologicalSort.sort(Arrays.asList(D, C, B, A));

        assertEquals(4, sorted.size());
        assertEquals(sorted.get(0), A);
        assertEquals(sorted.get(3), D);
    }

    @Test
    public void sortsDoubleDiamondGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex D = new TensorGaussianVertex(B, C);
        DoubleTensorVertex E = new TensorGaussianVertex(D, 1.0);
        DoubleTensorVertex F = new TensorGaussianVertex(D, 1.0);
        DoubleTensorVertex G = new TensorGaussianVertex(E, F);

        List<Vertex> vertices = Arrays.asList(F, B, G, C, E, A);
        List<? extends Vertex> sorted = TopologicalSort.sort(vertices);

        assertEquals(sorted.size(), vertices.size());
        assertEquals(sorted.get(0), A);
        assertEquals(sorted.get(5), G);
    }

    @Test
    public void sortsComplexGraph() {
        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, B);
        DoubleTensorVertex D = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex F = new TensorGaussianVertex(C, D);
        DoubleTensorVertex E = new TensorGaussianVertex(C, F);
        DoubleTensorVertex H = new TensorGaussianVertex(E, F);
        DoubleTensorVertex G = new TensorGaussianVertex(E, 1.0);
        DoubleTensorVertex I = new TensorGaussianVertex(H, G);
        DoubleTensorVertex J = new TensorGaussianVertex(H, 1.0);
        DoubleTensorVertex K = new TensorGaussianVertex(H, J);
        DoubleTensorVertex M = new TensorGaussianVertex(J, 1.0);
        DoubleTensorVertex L = new TensorGaussianVertex(K, H);

        List<Vertex> vertices = Arrays.asList(H, B, L, I, F, E, A, G, D, J, K, M, C);
        List<? extends Vertex> sorted = TopologicalSort.sort(vertices);

        assertEquals(sorted.size(), vertices.size());
        assertEquals(sorted.get(3), C);
        assertEquals(sorted.get(4), F);
        assertEquals(sorted.get(5), E);
    }

    private void assertExactOrder(List<? extends Vertex<?>> expected, List<? extends Vertex> actual) {
        assertEquals(actual.size(), expected.size());
        for (int i = 0; i < actual.size(); i++) {
            assertEquals(actual.get(i), expected.get(i));
        }
    }
}
