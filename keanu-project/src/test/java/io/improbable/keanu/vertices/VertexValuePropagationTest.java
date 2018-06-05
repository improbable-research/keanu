package io.improbable.keanu.vertices;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

import static io.improbable.keanu.vertices.TestGraphGenerator.*;
import static org.junit.Assert.assertEquals;

public class VertexValuePropagationTest {

    private final Logger log = LoggerFactory.getLogger(VertexValuePropagationTest.class);

    @Test
    public void doesNotDoUnnecessaryOperations() {

        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleTensorVertex start = new TensorGaussianVertex(0, 1);

        int links = 20;
        DoubleTensorVertex end = addLinks(start, n, m, links);

        start.setAndCascade(2.0);

        //Calculates the correct answer
        assertEquals(Math.pow(2, links + 1), end.getValue().scalar(), 0.0);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVertices() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleTensorVertex start = new TensorGaussianVertex(0, 1);

        DoubleTensorVertex end = addLinks(start, n, m, 1);

        DoubleTensorVertex nextLayerStart = new TensorGaussianVertex(end, 1);

        DoubleTensorVertex secondLayerEnd = addLinks(nextLayerStart, n, m, 1);

        start.setAndCascade(3.0);

        //Calculates the correct answer
        assertEquals(6.0, end.getValue().scalar(), 0.0);

        //Does the right amount of work
        assertEquals(3, n.get());
    }

    @Test
    public void doesPropagateAroundProbabilisticVertices() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleTensorVertex firstLayerStart = new TensorGaussianVertex(0, 1);

        DoubleTensorVertex firstLayerEnd = addLinks(firstLayerStart, n, m, 1);

        DoubleTensorVertex secondLayerStart = new TensorGaussianVertex(firstLayerEnd, 1);

        DoubleTensorVertex secondLayerLeft = sumVertex(secondLayerStart, firstLayerEnd, n, m, id -> log.info("OP on id: " + id));
        DoubleTensorVertex secondLayerRight = passThroughVertex(secondLayerStart, n, m, id -> log.info("OP on id: " + id));
        DoubleTensorVertex secondLayerEnd = sumVertex(secondLayerLeft, secondLayerRight, n, m, id -> log.info("OP on id: " + id));

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
