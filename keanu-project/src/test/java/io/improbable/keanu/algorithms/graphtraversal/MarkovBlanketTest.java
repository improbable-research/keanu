package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MarkovBlanketTest {

    @Test
    public void findBlanketFromSimpleGraph() {

        DoubleVertex A = new GaussianVertex(5.0, 1.0);
        DoubleVertex B = new GaussianVertex(5.0, 1.0);
        DoubleVertex C = new GaussianVertex(A, B);

        Set<Vertex> blanket = MarkovBlanket.get(C);

        assertTrue(blanket.size() == 2);
        assertTrue(blanket.containsAll(Arrays.asList(A, B)));
    }

    @Test
    public void findBlanketFromDoubleDiamondWithDeterministicGraph() {

        DoubleVertex A = new GaussianVertex(5.0, 1.0);
        DoubleVertex B = new GaussianVertex(A, 1.0);
        DoubleVertex C = new GaussianVertex(A, 1.0);
        DoubleVertex D = new GaussianVertex(B, C);
        DoubleVertex E = D.multiply(2.0);
        DoubleVertex F = new GaussianVertex(D, 1.0);
        DoubleVertex G = new GaussianVertex(E, F);

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, G)));
    }

    @Test
    public void findBlanketFromDoubleDiamondGraph() {

        DoubleVertex A = new GaussianVertex(5.0, 1.0);
        DoubleVertex B = new GaussianVertex(A, 1.0);
        DoubleVertex C = new GaussianVertex(A, 1.0);
        DoubleVertex D = new GaussianVertex(B, C);
        DoubleVertex E = new GaussianVertex(D, 1.0);
        DoubleVertex F = new GaussianVertex(D, 1.0);
        DoubleVertex G = new GaussianVertex(E, F);

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, E)));
    }

}
