package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

import static io.improbable.keanu.vertices.TestGraphGenerator.addLinks;
import static org.junit.Assert.assertEquals;

public class LazyEvalPropagationTest {

    private final Logger log = LoggerFactory.getLogger(LazyEvalPropagationTest.class);

    @Test
    public void doesNotDoUnnecessaryOperations() {

        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = new FloorVertex(ConstantVertex.of(4.2));

        int links = 20;
        DoubleVertex end = addLinks(start, n, m, links);

        end.lazyEval();

        //Value at the start has been evaluated correctly
        assertEquals(4.0, start.getValue().scalar(), 0.001);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVertices() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1);

        DoubleVertex end = addLinks(start, n, m, 1);

        DoubleVertex nextLayerStart = new GaussianVertex(end, 1);

        DoubleVertex secondLayerEnd = addLinks(nextLayerStart, n, m, 1);

        //Before lazy eval is called
        assertEquals(0, n.get());

        secondLayerEnd.lazyEval();

        //Lazy eval the additional 3 vertices at the end of the chain
        assertEquals(6, n.get());
    }

    @Test
    public void doesNotDoUnnecessaryOperationsOnVerticesThatShareParents() {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);

        DoubleVertex start1 = ConstantVertex.of(5.0);
        DoubleVertex start2 = ConstantVertex.of(5.0);
        DoubleVertex start3 = ConstantVertex.of(5.0);

        //start 2 is a shared parent between these sums
        DoubleVertex middleSum1 = TestGraphGenerator.sumVertex(start1, start2, n, m, id -> log.info("OP on id:" + id));
        DoubleVertex middleSum2 = TestGraphGenerator.sumVertex(start2, start3, n, m, id -> log.info("OP on id:" + id));

        DoubleVertex finalSum = TestGraphGenerator.sumVertex(middleSum1, middleSum2, n, m, id -> log.info("OP on id:" + id));

        finalSum.lazyEval();

        assertEquals(3, n.get());
    }
}
