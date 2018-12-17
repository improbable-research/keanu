package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.Test;

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

        Set<Vertex> verticesDepthFirst = LambdaSection.getVerticesDepthFirst(A, (v) -> {
            callsToNext.getAndIncrement();
            return v.getChildren();
        }, (v) -> {
            callsToPredicate.getAndIncrement();
            return true;
        });

        assertEquals(4, verticesDepthFirst.size());
        assertThat(verticesDepthFirst, containsInAnyOrder(A, sinA, APlusSinA, C));

        assertEquals(3, callsToNext.intValue());
        assertEquals(3, callsToPredicate.intValue());
    }
}
