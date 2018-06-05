package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.TensorDoubleBinaryOpLambda;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.TensorDoubleUnaryOpLambda;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class TestGraphGenerator {

    private static final Logger log = LoggerFactory.getLogger(TestGraphGenerator.class);

    static DoubleTensorVertex addLinks(DoubleTensorVertex end, AtomicInteger opCount, AtomicInteger dualNumberCount, int links) {

        for (int i = 0; i < links; i++) {
            DoubleTensorVertex left = passThroughVertex(end, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
            DoubleTensorVertex right = passThroughVertex(end, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
            end = sumVertex(left, right, opCount, dualNumberCount, id -> log.info("OP on id:" + id));
        }

        return end;
    }

    static DoubleTensorVertex passThroughVertex(DoubleTensorVertex from, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        final long id = Vertex.ID_GENERATOR.get();
        return new TensorDoubleUnaryOpLambda<>(from, (a) -> {
            opCount.incrementAndGet();
            onOp.accept(id);
            return a;
        }, (a) -> {
            dualNumberCount.incrementAndGet();
            return a.get(from);
        });
    }

    static DoubleTensorVertex sumVertex(DoubleTensorVertex left, DoubleTensorVertex right, AtomicInteger opCount, AtomicInteger dualNumberCount, Consumer<Long> onOp) {
        final long id = Vertex.ID_GENERATOR.get();
        return new TensorDoubleBinaryOpLambda<>(left, right, (a, b) -> {
            opCount.incrementAndGet();
            onOp.accept(id);
            return a.plus(b);
        }, (a) -> {
            dualNumberCount.incrementAndGet();
            return a.get(left).add(a.get(right));
        });
    }

}
