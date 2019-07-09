package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.Test;

import java.util.Arrays;
import java.util.Set;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class LambdaSectionEfficiencyTest {

    @Test
    public void doesNotVisitVerticesMoreThanOnce() {
        GaussianVertex A = new GaussianVertex(0, 1);
        SinVertex sinA = A.sin();
        AdditionVertex APlusSinA = A.plus(sinA);
        GaussianVertex C = new GaussianVertex(APlusSinA, 1);

        MutableInt callsToNext = new MutableInt(0);
        MutableInt callsToPredicate = new MutableInt(0);

        Set<Vertex> verticesDepthFirst = Propagation.getVertices(A, (v) -> {
                callsToNext.increment();
                return v.getChildren();
            }, (v) -> v.isProbabilistic() || v.isObserved(),
            (v) -> {
                callsToPredicate.increment();
                return true;
            });

        assertEquals(4, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, sinA, APlusSinA, C));

        assertEquals(3, callsToNext.intValue());
        assertEquals(3, callsToPredicate.intValue());
    }

    @Test
    public void doesNotVisitVerticesMoreThanOnceForCollection() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(A, 1);
        DoubleVertex BSin = B.sin();
        DoubleVertex BCos = B.cos();
        GaussianVertex C = new GaussianVertex(BCos, 1);
        GaussianVertex D = new GaussianVertex(C, BSin);

        MutableInt callsToNext = new MutableInt(0);
        MutableInt callsToPredicate = new MutableInt(0);

        Set<Vertex> verticesDepthFirst = Propagation.getVertices(Arrays.asList(A, B, C), (v) -> {
                callsToNext.increment();
                return v.getChildren();
            },
            (v) -> v.isProbabilistic() || v.isObserved(),
            (v) -> {
                callsToPredicate.increment();
                return true;
            });

        assertEquals(6, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, B, BSin, BCos, C, D));

        assertEquals(5, callsToNext.intValue());
        assertEquals(4, callsToPredicate.intValue());
    }
}
