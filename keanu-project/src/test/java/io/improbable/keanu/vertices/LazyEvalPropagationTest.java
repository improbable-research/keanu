package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.TensorFloorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
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
        DoubleTensorVertex start = new TensorFloorVertex(new ConstantDoubleTensorVertex(4.2));

        int links = 20;
        DoubleTensorVertex end = addLinks(start, n, m, links);

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
        DoubleTensorVertex start = new TensorGaussianVertex(0, 1);

        DoubleTensorVertex end = addLinks(start, n, m, 1);

        DoubleTensorVertex nextLayerStart = new TensorGaussianVertex(end, 1);

        DoubleTensorVertex secondLayerEnd = addLinks(nextLayerStart, n, m, 1);

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

        DoubleTensorVertex start1 = new ConstantDoubleTensorVertex(5.0);
        DoubleTensorVertex start2 = new ConstantDoubleTensorVertex(5.0);
        DoubleTensorVertex start3 = new ConstantDoubleTensorVertex(5.0);

        //start 2 is a shared parent between these sums
        DoubleTensorVertex middleSum1 = TestGraphGenerator.sumVertex(start1, start2, n, m, id -> log.info("OP on id:" + id));
        DoubleTensorVertex middleSum2 = TestGraphGenerator.sumVertex(start2, start3, n, m, id -> log.info("OP on id:" + id));

        DoubleTensorVertex finalSum = TestGraphGenerator.sumVertex(middleSum1, middleSum2, n, m, id -> log.info("OP on id:" + id));

        finalSum.lazyEval();

        assertEquals(3, n.get());
    }
}
