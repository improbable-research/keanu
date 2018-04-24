package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;

public class LazyEvalPropagationTest {

    private final Logger log = LoggerFactory.getLogger(LazyEvalPropagationTest.class);

    Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void doesNotDoUnnecessaryOperations() {

        AtomicInteger n = new AtomicInteger(0);
        DoubleVertex start = new FloorVertex(4.2);

        int links = 20;
        DoubleVertex end = addLinks(start, n, links);

        end.lazyEval();

        //Value at the start has been evaluated correctly
        assertEquals(4.0, start.getValue(), 0.001);

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

        //Before lazy eval is called
        assertEquals(0, n.get());

        secondLayerEnd.lazyEval();

        //Lazy eval the additional 3 vertices at the end of the chain
        assertEquals(6, n.get());
    }

    @Test
    public void doesNotDoUnnecessaryOperationsOnVerticesThatShareParents() {
        AtomicInteger n = new AtomicInteger(0);

        DoubleVertex start1 = new ConstantDoubleVertex(5.0);
        DoubleVertex start2 = new ConstantDoubleVertex(5.0);
        DoubleVertex start3 = new ConstantDoubleVertex(5.0);

        //start 2 is a shared parent between these sums
        DoubleVertex middleSum1 = sumVertex(start1, start2, n, id -> log.info("OP on id:" + id));
        DoubleVertex middleSum2 = sumVertex(start2, start3, n, id -> log.info("OP on id:" + id));

        DoubleVertex finalSum = sumVertex(middleSum1, middleSum2, n, id -> log.info("OP on id:" + id));

        finalSum.lazyEval();

        assertEquals(3, n.get());
    }

    private DoubleVertex addLinks(DoubleVertex end, AtomicInteger n, int links) {

        for (int i = 0; i < links; i++) {
            DoubleVertex left = passThroughVertex(end, n, id -> log.info("OP on id:" + id));
            DoubleVertex right = passThroughVertex(end, n, id -> log.info("OP on id:" + id));
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
