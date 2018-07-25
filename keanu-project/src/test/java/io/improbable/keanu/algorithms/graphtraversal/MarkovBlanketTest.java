package io.improbable.keanu.algorithms.graphtraversal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Set;

import org.junit.Test;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class MarkovBlanketTest {

    @Test
    public void findBlanketFromSimpleGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex C = VertexOfType.gaussian(A, B);

        Set<Vertex> blanket = MarkovBlanket.get(C);

        assertTrue(blanket.size() == 2);
        assertTrue(blanket.containsAll(Arrays.asList(A, B)));
    }

    @Test
    public void findBlanketFromDoubleDiamondWithDeterministicGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex C = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex D = VertexOfType.gaussian(B, C);
        DoubleVertex E = D.multiply(2.0);
        DoubleVertex F = VertexOfType.gaussian(D, ConstantVertex.of(1.0));
        DoubleVertex G = VertexOfType.gaussian(E, F);

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, G)));
    }

    @Test
    public void findBlanketFromDoubleDiamondGraph() {

        DoubleVertex A = VertexOfType.gaussian(5.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex C = VertexOfType.gaussian(A, ConstantVertex.of(1.0));
        DoubleVertex D = VertexOfType.gaussian(B, C);
        DoubleVertex E = VertexOfType.gaussian(D, ConstantVertex.of(1.0));
        DoubleVertex F = VertexOfType.gaussian(D, ConstantVertex.of(1.0));

        Set<Vertex> blanket = MarkovBlanket.get(D);

        assertEquals(4, blanket.size());
        assertTrue(blanket.containsAll(Arrays.asList(B, C, F, E)));
    }

}
