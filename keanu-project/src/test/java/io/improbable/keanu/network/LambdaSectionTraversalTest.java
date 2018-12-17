package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Set;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class LambdaSectionTraversalTest {

    GaussianVertex A;
    GaussianVertex B;
    AdditionVertex aPlusB;
    ConstantDoubleVertex cSigma;
    GaussianVertex C;

    @Before
    public void setup() {
        A = new GaussianVertex(0, 1);
        B = new GaussianVertex(0, 1);
        aPlusB = A.plus(B);
        cSigma = ConstantVertex.of(1.0);
        C = new GaussianVertex(aPlusB, cSigma);
    }

    @Test
    public void doesGetDownstreamProbabilisticVertices() {

        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, false);
        Set<Vertex> verticesDepthFirst = lambdaSection.getLatentAndObservedVertices();

        assertEquals(2, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, C));
    }

    @Test
    public void doesGetAllDownstreamVertices() {

        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, true);
        Set<Vertex> verticesDepthFirst = lambdaSection.getAllVertices();

        assertEquals(3, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, aPlusB, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVertices() {

        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, false);
        Set<Vertex> verticesDepthFirst = lambdaSection.getLatentAndObservedVertices();

        assertEquals(3, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetAllUpstreamVertices() {

        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, true);
        Set<Vertex> verticesDepthFirst = lambdaSection.getAllVertices();

        assertEquals(5, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, B, aPlusB, cSigma, C));
    }
}
