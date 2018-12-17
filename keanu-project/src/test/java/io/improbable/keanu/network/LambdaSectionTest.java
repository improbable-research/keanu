package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class LambdaSectionTest {

    @Test
    public void doesGetDownstreamProbabilisticVertices() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        GaussianVertex C = new GaussianVertex(A.plus(B), 1);

        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, false);
        Set<Vertex> verticesDepthFirst = lambdaSection.getLatentAndObservedVertices();

        assertEquals(2, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, C));
    }

    @Test
    public void doesGetAllDownstreamVertices() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        AdditionVertex plus = A.plus(B);
        GaussianVertex C = new GaussianVertex(plus, 1);

        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, true);
        Set<Vertex> verticesDepthFirst = lambdaSection.getAllVertices();

        assertEquals(3, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, plus, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVertices() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        GaussianVertex C = new GaussianVertex(A.plus(B), 1);

        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, false);
        Set<Vertex> verticesDepthFirst = lambdaSection.getLatentAndObservedVertices();

        assertEquals(3, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetAllUpstreamVertices() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        AdditionVertex plus = A.plus(B);
        ConstantDoubleVertex sigma = ConstantVertex.of(1.0);
        GaussianVertex C = new GaussianVertex(plus, sigma);

        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, true);
        Set<Vertex> verticesDepthFirst = lambdaSection.getAllVertices();

        assertEquals(5, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, B, plus, sigma, C));
    }

    @Test
    public void doesNotVisitVerticesMoreThanOnce() {
        GaussianVertex A = new GaussianVertex(0, 1);
        SinVertex sinA = A.sin();
        AdditionVertex APlusSinA = A.plus(sinA);
        GaussianVertex C = new GaussianVertex(APlusSinA, 1);

        AtomicInteger callsToNext = new AtomicInteger(0);
        AtomicInteger callsToPredicate = new AtomicInteger(0);

        Set<Vertex> verticesDepthFirst = LambdaSection.getVerticesDepthFirst(A, (v) -> {
            callsToNext.getAndIncrement();
            return v.getChildren();
        }, (v) -> {
            callsToPredicate.getAndIncrement();
            return true;
        });

        assertEquals(4, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, sinA, APlusSinA, C));

        assertEquals(3, callsToNext.get());
        assertEquals(3, callsToPredicate.get());
    }
}
