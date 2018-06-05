package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MarkovBlanketTest {

    @Test
    public void findBlanketFromSimpleGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, B);

        Set<Vertex> blanket = MarkovBlanket.get(C);

        assertTrue(blanket.size() == 2);
        assertTrue(blanket.containsAll(Arrays.asList(A, B)));
    }

    @Test
    public void findBlanketFromDoubleDiamondWithDeterministicGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex D = new TensorGaussianVertex(B, C);
        DoubleTensorVertex E = D.multiply(2.0);
        DoubleTensorVertex F = new TensorGaussianVertex(D, 1.0);
        DoubleTensorVertex G = new TensorGaussianVertex(E, F);

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, G)));
    }

    @Test
    public void findBlanketFromDoubleDiamondGraph() {

        DoubleTensorVertex A = new TensorGaussianVertex(5.0, 1.0);
        DoubleTensorVertex B = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex C = new TensorGaussianVertex(A, 1.0);
        DoubleTensorVertex D = new TensorGaussianVertex(B, C);
        DoubleTensorVertex E = new TensorGaussianVertex(D, 1.0);
        DoubleTensorVertex F = new TensorGaussianVertex(D, 1.0);
        DoubleTensorVertex G = new TensorGaussianVertex(E, F);

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, E)));
    }

}
