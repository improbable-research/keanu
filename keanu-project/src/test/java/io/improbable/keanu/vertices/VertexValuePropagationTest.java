package io.improbable.keanu.vertices;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.TestGraphGenerator.addLinks;
import static io.improbable.keanu.vertices.TestGraphGenerator.passThroughVertex;
import static io.improbable.keanu.vertices.TestGraphGenerator.sumVertex;

import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class VertexValuePropagationTest {

    private final Logger log = LoggerFactory.getLogger(VertexValuePropagationTest.class);

    @Test
    public void doesNotDoUnnecessaryOperationsOnCascade() {

        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = VertexOfType.gaussian(0., 1.);

        int links = 20;
        DoubleVertex end = addLinks(start, n, m, links);

        start.setAndCascade(2.0);

        //Calculates the correct answer
        assertEquals(Math.pow(2, links + 1), end.getValue().scalar(), 0.0);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVerticesOnCascade() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = VertexOfType.gaussian(0., 1.);

        DoubleVertex end = addLinks(start, n, m, 1);

        DoubleVertex nextLayerStart = VertexOfType.gaussian(end, ConstantVertex.of(1.0));

        DoubleVertex secondLayerEnd = addLinks(nextLayerStart, n, m, 1);

        start.setAndCascade(3.0);

        //Calculates the correct answer
        assertEquals(6.0, end.getValue().scalar(), 0.0);

        //Does the right amount of work
        assertEquals(3, n.get());
    }

    @Test
    public void doesPropagateAroundProbabilisticVerticesOnCascade() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex firstLayerStart = VertexOfType.gaussian(0., 1.);

        DoubleVertex firstLayerEnd = addLinks(firstLayerStart, n, m, 1);

        DoubleVertex secondLayerStart = VertexOfType.gaussian(firstLayerEnd, ConstantVertex.of(1.));

        DoubleVertex secondLayerLeft = sumVertex(secondLayerStart, firstLayerEnd, n, m, id -> log.info("OP on id: " + id));
        DoubleVertex secondLayerRight = passThroughVertex(secondLayerStart, n, m, id -> log.info("OP on id: " + id));
        DoubleVertex secondLayerEnd = sumVertex(secondLayerLeft, secondLayerRight, n, m, id -> log.info("OP on id: " + id));

        secondLayerStart.setValue(2.0);
        firstLayerStart.setValue(3.0);
        VertexValuePropagation.cascadeUpdate(firstLayerStart, secondLayerStart);

        //Calculates the correct answer
        assertEquals(6.0, firstLayerEnd.getValue().scalar(), 0.0);
        assertEquals(10.0, secondLayerEnd.getValue().scalar(), 0.0);

        //Does the right amount of work
        assertEquals(6, n.get());
    }

}
