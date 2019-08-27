package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static io.improbable.keanu.vertices.TestGraphGenerator.addLinks;
import static org.junit.Assert.assertEquals;

@Slf4j
public class EvalPropagationTest {

    @Test
    public void doesNotDoUnnecessaryOperationsOnEval() {
        assertDoesNotDoUnnecessaryOperations(Vertex::eval);
    }

    @Test
    public void doesNotDoUnnecessaryOperationsOnLazyEval() {
        assertDoesNotDoUnnecessaryOperations(Vertex::lazyEval);
    }

    private void assertDoesNotDoUnnecessaryOperations(Consumer<Vertex> evalFunction) {

        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = ConstantVertex.of(4.2).floor();

        int links = 20;
        DoubleVertex end = addLinks(start, n, m, links);

        evalFunction.accept(end);

        //Value at the start has been evaluated correctly
        assertEquals(4.0, start.getValue().scalar(), 0.001);

        //Does the right amount of work
        assertEquals(3 * links, n.get());
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVerticesOnEval() {
        assertDoesNotPropagateThroughProbabilisticVertices(Vertex::eval);
    }

    @Test
    public void doesNotPropagateThroughProbabilisticVerticesOnLazyEval() {
        assertDoesNotPropagateThroughProbabilisticVertices(Vertex::lazyEval);
    }

    private void assertDoesNotPropagateThroughProbabilisticVertices(Consumer<Vertex> evalFunction) {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);
        DoubleVertex start = new GaussianVertex(0, 1);

        DoubleVertex end = addLinks(start, n, m, 1);

        DoubleVertex nextLayerStart = new GaussianVertex(end, 1);

        DoubleVertex secondLayerEnd = addLinks(nextLayerStart, n, m, 1);

        //Before lazy eval is called
        assertEquals(0, n.get());

        evalFunction.accept(secondLayerEnd);

        //Lazy eval the additional 3 vertices at the end of the chain
        assertEquals(6, n.get());
    }

    @Test
    public void doesNotDoUnnecessaryOperationsOnVerticesThatShareParentsOnEval() {
        assertDoesNotDoUnnecessaryOperationsOnVerticesThatShareParents(Vertex::eval);
    }

    @Test
    public void doesNotDoUnnecessaryOperationsOnVerticesThatShareParentsOnLazyEval() {
        assertDoesNotDoUnnecessaryOperationsOnVerticesThatShareParents(Vertex::lazyEval);
    }

    private void assertDoesNotDoUnnecessaryOperationsOnVerticesThatShareParents(Consumer<Vertex> evalFunction) {
        AtomicInteger n = new AtomicInteger(0);
        AtomicInteger m = new AtomicInteger(0);

        DoubleVertex start1 = ConstantVertex.of(5.0);
        DoubleVertex start2 = ConstantVertex.of(5.0);
        DoubleVertex start3 = ConstantVertex.of(5.0);

        //start 2 is a shared parent between these sums
        DoubleVertex middleSum1 = TestGraphGenerator.sumVertex(start1, start2, n, m, id -> log.info("OP on id:" + id));
        DoubleVertex middleSum2 = TestGraphGenerator.sumVertex(start2, start3, n, m, id -> log.info("OP on id:" + id));

        DoubleVertex finalSum = TestGraphGenerator.sumVertex(middleSum1, middleSum2, n, m, id -> log.info("OP on id:" + id));

        evalFunction.accept(finalSum);

        assertEquals(3, n.get());
    }

    static class BlackBoxVertex extends DoubleUnaryOpVertex implements Differentiable, NonSaveableVertex {

        private final AtomicInteger n;

        public BlackBoxVertex(DoubleVertex inputVertex, AtomicInteger n) {
            super(inputVertex);
            this.n = n;
        }

        @Override
        protected DoubleTensor op(DoubleTensor value) {
            n.incrementAndGet();
            return DoubleTensor.create(new double[]{0, 1, 2});
        }

        @Override
        public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
            return null;
        }
    }

    @Test
    public void doesNotRedoWorkAlreadyDoneOnLazyEval() {
        AtomicInteger n = new AtomicInteger(0);

        DoubleVertex start = new GaussianVertex(0, 1);

        DoubleVertex blackBox = new BlackBoxVertex(start, n);

        DoubleVertex pluck0 = blackBox.lambda(blackBox.getShape(), bb -> DoubleTensor.scalar(bb.getValue(0)), null, null);
        DoubleVertex pluck1 = blackBox.lambda(blackBox.getShape(), bb -> DoubleTensor.scalar(bb.getValue(1)), null, null);
        DoubleVertex pluck2 = blackBox.lambda(blackBox.getShape(), bb -> DoubleTensor.scalar(bb.getValue(2)), null, null);

        pluck0.lazyEval();
        pluck1.lazyEval();
        pluck2.lazyEval();

        assertEquals(1, n.get());
        assertEquals(0, pluck0.getValue(0), 0.0);
        assertEquals(1, pluck1.getValue(0), 0.0);
        assertEquals(2, pluck2.getValue(0), 0.0);
    }
}
