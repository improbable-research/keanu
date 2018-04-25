package io.improbable.keanu.vertices;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;

public class VertexValuePropagationTest {

    private final Logger log = LoggerFactory.getLogger(VertexValuePropagationTest.class);

    Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void doesNotDoUnnecessaryOperations() {

        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1, random);

        int links = 20;
        DoubleVertex end = addLinks(start, n, links);

        start.setAndCascade(2.0);

        //Calculates the correct answer
        assertEquals(Math.pow(2, links + 1), end.getValue(), 0.0);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVertices() {
        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1, random);

        DoubleVertex end = addLinks(start, n, 1);

        DoubleVertex nextLayerStart = new GaussianVertex(end, 1, random);

        DoubleVertex secondLayerEnd = addLinks(nextLayerStart, n, 1);

        start.setAndCascade(3.0);

        //Calculates the correct answer
        assertEquals(6.0, end.getValue(), 0.0);

        //Does the right amount of work
        assertEquals(3, n.get());
    }

    @Test
    public void doesPropagateAroundProbabilisticVertices() {
        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex firstLayerStart = new GaussianVertex(0, 1, random);

        DoubleVertex firstLayerEnd = addLinks(firstLayerStart, n, 1);

        DoubleVertex secondLayerStart = new GaussianVertex(firstLayerEnd, 1, random);

        DoubleVertex secondLayerLeft = sumVertex(secondLayerStart, firstLayerEnd, n, id -> log.info("OP on id: " + id));
        DoubleVertex secondLayerRight = passThroughVertex(secondLayerStart, n, id -> log.info("OP on id: " + id));
        DoubleVertex secondLayerEnd = sumVertex(secondLayerLeft, secondLayerRight, n, id -> log.info("OP on id: " + id));

        secondLayerStart.setValue(2.0);
        firstLayerStart.setValue(3.0);
        VertexValuePropagation.cascadeUpdate(firstLayerStart, secondLayerStart);

        //Calculates the correct answer
        assertEquals(6.0, firstLayerEnd.getValue(), 0.0);
        assertEquals(10.0, secondLayerEnd.getValue(), 0.0);

        //Does the right amount of work
        assertEquals(6, n.get());
    }

    private DoubleVertex addLinks(DoubleVertex end, AtomicInteger n, int links) {

        for (int i = 0; i < links; i++) {
            DoubleVertex left = passThroughVertex(end, n, id -> log.info("OP on id: " + id));
            DoubleVertex right = passThroughVertex(end, n, id -> log.info("OP on id: " + id));
            end = sumVertex(left, right, n, id -> log.info("OP on id:" + id));
        }

        return end;
    }

    private DoubleVertex passThroughVertex(DoubleVertex from, AtomicInteger n, Consumer<Long> onOp) {
        final long id = Vertex.idGenerator.get();
        return new DoubleUnaryOpLambda<>(from, (a) -> {
            n.incrementAndGet();
            onOp.accept(id);
            return a;
        });
    }

    private DoubleVertex sumVertex(DoubleVertex left, DoubleVertex right, AtomicInteger n, Consumer<Long> onOp) {
        final long id = Vertex.idGenerator.get();
        return new DoubleBinaryOpLambda<>(left, right, (a, b) -> {
            n.incrementAndGet();
            onOp.accept(id);
            return a + b;
        });
    }
}
